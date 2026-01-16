from egomimic.utils.aws.aws_sql import (
    episode_table_to_df,
    create_default_engine,
)
from tqdm import tqdm
from pathlib import PurePosixPath

def rename_processed_to_episode_hash(df, dry_run, max_workers=16):
    """
    For each row in the sql table there's processed path of the form rldb:/mecka/flagship/692ea0262fa9ba56c08f8097/	
    We want to change this to s3://rldb/mecka/flagship/<row.episode_hash>/
    There are a lot of rows (42k), ideally the file renaming should be doen in a batch fashion.
    The processed paths are all on S3.
    args:
        df: pandas dataframe of the episode sql table
        dry_run: if True, only print what would be done
        max_workers: number of threads used for S3 copy operations
    """
    import pandas as pd
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
    from boto3.s3.transfer import TransferConfig, S3Transfer
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sqlalchemy import MetaData, Table, bindparam, update
    from egomimic.utils.aws.aws_sql import create_default_engine
    def parse_s3_uri(uri):
        if uri is None:
            return None, None
        uri = str(uri)
        if not uri.startswith("s3://"):
            return None, None
        rest = uri[len("s3://"):]
        bucket, _, key = rest.partition("/")
        return bucket, key
    
    def parse_rldb_uri(uri):
        """
        Given a URI of the form rldb:/mecka/flagship/692ea0262fa9ba56c08f8097/
        return bucket rldb and key mecka/flagship/692ea0262fa9ba56c08f8097/
        """
        if uri is None:
            return None, None
        uri = str(uri)
        if not uri.startswith("rldb:/"):
            return None, None
        rest = uri[len("rldb:/"):]
        bucket = "rldb"
        key = rest.lstrip("/")
        return bucket, key

    def ensure_trailing_slash(prefix):
        if prefix and not prefix.endswith("/"):
            return prefix + "/"
        return prefix

    def normalize_key_prefix(prefix):
        if not prefix:
            return ""
        text = str(prefix).strip().strip("/")
        return PurePosixPath(text).as_posix()

    def chunked(items, size):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def move_s3(pairs, s3_client, transfer, chunk_size=100, max_workers=16):
        totals = {"prefixes": 0, "objects": 0, "failures": 0}
        # tqdm is optional; fall back to no-op iterators if not installed
        # prefix_bar = tqdm(total=len(pairs), desc="Prefixes", unit="prefix", leave=True)
        # object_bar = tqdm(total=0, desc="Objects", unit="obj", leave=False)
        for batch in tqdm(chunked(pairs, chunk_size), desc="Batches", unit="batch"):
            for old_uri, new_uri in batch:
                old_bucket, old_prefix = parse_rldb_uri(old_uri)
                new_bucket, new_prefix = parse_s3_uri(new_uri)
                # print(f"Processing: {old_uri} -> {new_uri}.  {old_bucket}, {old_prefix} -> {new_bucket}, {new_prefix}")

                if not old_bucket or not new_bucket:
                    totals["failures"] += 1
                    # prefix_bar.update(1)
                    continue

                if dry_run:
                    print(
                        f"[DRY RUN] s3://{old_bucket}/{old_prefix} -> s3://{new_bucket}/{new_prefix}"
                    )

                paginator = s3_client.get_paginator("list_objects_v2")
                page_iter = paginator.paginate(Bucket=old_bucket, Prefix=old_prefix)
                keys = []
                for page in page_iter:
                    for obj in page.get("Contents", []):
                        keys.append(obj["Key"])

                if not keys:
                    if dry_run:
                        print(f"[DRY RUN] No objects under s3://{old_bucket}/{old_prefix}")
                    # prefix_bar.update(1)
                    continue

                totals["prefixes"] += 1
                totals["objects"] += len(keys)
                # object_bar.total += len(keys)
                # object_bar.refresh()

                if dry_run:
                    # object_bar.update(len(keys))
                    # prefix_bar.update(1)
                    continue

                to_delete = []
                futures = []
                for key in keys:
                    new_key = new_prefix + key[len(old_prefix):]
                    copy_source = {"Bucket": old_bucket, "Key": key}
                    try:
                        futures.append(
                            (key, new_key, transfer.copy(copy_source, new_bucket, new_key))
                        )
                    except ClientError as exc:
                        totals["failures"] += 1
                        print(
                            "Failed to start copy "
                            f"s3://{old_bucket}/{key} -> s3://{new_bucket}/{new_key}: {exc}"
                        )

                for key, new_key, fut in futures:
                    try:
                        fut.result()
                        to_delete.append({"Key": key})
                    except ClientError as exc:
                        totals["failures"] += 1
                        print(
                            "Failed to copy "
                            f"s3://{old_bucket}/{key} -> s3://{new_bucket}/{new_key}: {exc}"
                        )
                    # object_bar.update(1)

                for chunk in chunked(to_delete, 1000):
                    try:
                        s3_client.delete_objects(Bucket=old_bucket, Delete={"Objects": chunk})
                    except ClientError as exc:
                        totals["failures"] += len(chunk)
                        print(
                            "Failed to delete objects under "
                            f"s3://{old_bucket}/{old_prefix}: {exc}"
                        )

                # prefix_bar.update(1)
        # prefix_bar.close()
        # object_bar.close()
        return totals

    if "processed_path" not in df.columns or "episode_hash" not in df.columns:
        raise ValueError("df must include 'processed_path' and 'episode_hash' columns")

    s3 = boto3.client(
        "s3",
        config=Config(retries={"max_attempts": 20, "mode": "adaptive"}, max_pool_connections=128),
    )
    transfer = S3Transfer(
        s3,
        config=TransferConfig(
            max_concurrency=max_workers,
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            use_threads=True,
        ),
    )
    pairs = []
    updates = []
    skipped = 0

    for _, row in df.iterrows():
        old_bucket, old_prefix = parse_rldb_uri(row.get("processed_path"))
        episode_hash = row.get("episode_hash")
        if not old_bucket or not old_prefix or not episode_hash:
            skipped += 1
            continue

        base_prefix = normalize_key_prefix(old_prefix)
        base_path = PurePosixPath(base_prefix)
        parent_path = base_path.parent if base_path.name else base_path
        new_key = parent_path / str(episode_hash)
        new_uri = f"s3://{old_bucket}/{ensure_trailing_slash(new_key.as_posix())}"

        pairs.append((row.get("processed_path"), new_uri))
        print(pairs[-1][0], "->", pairs[-1][1])
        updates.append({"episode_hash": episode_hash, "processed_path": new_uri})

    print(f"Got {len(pairs)} paths to move, {skipped} rows to skip.")
    move_stats = move_s3(pairs, s3, transfer, chunk_size=100, max_workers=max_workers)

    if dry_run:
        print(f"[DRY RUN] Would update {len(updates)} rows in app.episodes")
        return {
            "move": move_stats,
            "updates": len(updates),
            "skipped": skipped,
        }

    engine = create_default_engine()
    metadata = MetaData()
    episodes_tbl = Table("episodes", metadata, autoload_with=engine, schema="app")
    stmt = (
        update(episodes_tbl)
        .where(episodes_tbl.c.episode_hash == bindparam("episode_hash"))
        .values(processed_path=bindparam("processed_path"))
    )
    with engine.begin() as conn:
        conn.execute(stmt, updates)

    return {
        "move": move_stats,
        "updates": len(updates),
        "skipped": skipped,
    }



def main():
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    df = df[df["lab"] != "mecka"]
    print(len(df), "episodes to process")
    rename_processed_to_episode_hash(df, dry_run=True, max_workers=16)


main()

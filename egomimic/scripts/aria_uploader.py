import asyncio
from pathlib import Path

from abstract_upload import UploadTemplate


class VRSUpload(UploadTemplate):
    """VRS file uploader for ARIA embodiment."""
    
    def __init__(self):
        """
        Initialize VRS uploader for ARIA embodiment.
        Directory will be prompted from user during file discovery.
        """
        super().__init__("ARIA_TEST")
        self.datatype = ".vrs"
    
    def verifylocal(self):
        """
        Discover VRS files with their corresponding JSON companion files.
        Only processes files that have both .vrs and .vrs.json files present.
        Call set_directory() first to select the directory.
        """
        # Check if directory has been set
        if not self.directory_prompted or not self.local_dir:
            raise ValueError("Directory not set. Please call set_directory() first.")
            
        vrs_files = [
            file for file in self.local_dir.iterdir() 
            if file.suffix == self.datatype and file.is_file()
        ]
        
        for vrs_file in vrs_files:
            json_file = vrs_file.with_suffix(f"{self.datatype}.json")
            
            if json_file.exists() and json_file.is_file():
                self.file_paths.append((vrs_file, json_file))
                print(f"Found VRS pair: {vrs_file.name} + {json_file.name}")
            else:
                print(f"Missing JSON file for {vrs_file.name}, skipping.")
        
        print(f"\nDiscovered {len(self.file_paths)} valid VRS file pairs.")

    async def run(self):
        """
        Process and upload all discovered VRS file pairs.
        Collects metadata sequentially but uploads concurrently in background.
        """
        background_tasks = []
        
        for vrs_file, json_file in self.file_paths:
            # Collect metadata (requires user input)
            metadict, stamped_name = self.prompt_user(vrs_file)

            print(f"Starting background upload for {vrs_file.name}...")
            
            # Create upload coroutines for all files
            upload_coroutines = [
                self.upload_metadata(stamped_name, metadict),
                self.upload_file(vrs_file, stamped_name + vrs_file.suffix),
                self.upload_file(json_file, stamped_name + json_file.suffix)
            ]
            
            # Start upload task in background (non-blocking)
            async def upload_and_report(file_name, tasks):
                try:
                    await asyncio.gather(*tasks)
                    print(f"Completed upload for {file_name}")
                except Exception as e:
                    print(f"Failed to upload {file_name}: {e}")
                    raise
            
            task = asyncio.create_task(upload_and_report(vrs_file.name, upload_coroutines))
            background_tasks.append(task)
        
        # Wait for all background uploads to complete
        if background_tasks:
            print(f"\nWaiting for {len(background_tasks)} background upload sets to complete...")
            await asyncio.gather(*background_tasks)
            print("All uploads completed!")
    
def main():
    """Main entry point for VRS uploader."""
    # Initialize uploader
    uploader = VRSUpload()
    
    # Set directory first
    uploader.set_directory()
    
    # Verify files
    uploader.verifylocal()
    
    if not uploader.file_paths:
        print("No VRS file pairs found to upload.")
        return
    
    # Run the async upload process
    asyncio.run(uploader.run())


if __name__ == "__main__":
    main()
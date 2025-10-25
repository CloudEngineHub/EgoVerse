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
        super().__init__("aria")
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

    def run(self):
        """
        Process and upload all discovered HDF5 files.
        First collects all metadata from user, then uploads everything at once.
        """

        file_metadata_pairs = []

        for vrs_file, json_file in self.file_paths:
            metadict, stamped_name = self.prompt_user(vrs_file)
            file_metadata_pairs.append((vrs_file, json_file, metadict, stamped_name))

        for vrs_file, json_file, metadict, stamped_name in file_metadata_pairs:
            self.upload_metadata(stamped_name, metadict)
            self.upload_file(vrs_file, stamped_name + vrs_file.suffix)
            self.upload_file(json_file, stamped_name + json_file.suffix)
    
def main():
    """Main entry point for VRS uploader."""
    # Initialize uploader
    uploader = VRSUpload()
    
    # Set directory first
    uploader.set_directory()
    
    # Verify files
    uploader.verifylocal()

    uploader.run()

if __name__ == "__main__":
    main()
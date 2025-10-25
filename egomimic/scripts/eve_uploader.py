import asyncio
from pathlib import Path

from abstract_upload import UploadTemplate


class HDF5Upload(UploadTemplate):
    """HDF5 file uploader for EVE embodiment."""
    
    def __init__(self):
        """
        Initialize HDF5 uploader for EVE embodiment.
        Directory will be prompted from user during file discovery.
        """
        super().__init__("EVE_TEST")
        self.datatype = ".hdf5"
    
    def verifylocal(self):
        """
        Discover HDF5 files in the local directory.
        Adds all valid HDF5 files to the processing queue.
        Call set_directory() first to select the directory.
        """
        # Check if directory has been set
        if not self.directory_prompted or not self.local_dir:
            raise ValueError("Directory not set. Please call set_directory() first.")
            
        hdf5_files = [
            file for file in self.local_dir.iterdir() 
            if file.suffix == self.datatype and file.is_file()
        ]
        
        for hdf5_file in hdf5_files:
            self.file_paths.append(hdf5_file)
            print(f"Found HDF5 file: {hdf5_file.name}")
        
        print(f"\nDiscovered {len(self.file_paths)} HDF5 files to process.")

    async def run(self):
        """
        Process and upload all discovered HDF5 files.
        Collects metadata sequentially but uploads concurrently in background.
        """
        background_tasks = []
        
        for hdf5_file in self.file_paths:
            # Collect metadata (requires user input)
            metadict, stamped_name = self.prompt_user(hdf5_file)
            
            print(f"Starting background upload for {hdf5_file.name}...")

            # Create upload coroutines
            upload_coroutines = [
                self.upload_metadata(stamped_name, metadict),
                self.upload_file(hdf5_file, stamped_name + hdf5_file.suffix),
            ]
            
            # Start upload task in background (non-blocking)
            async def upload_and_report(file_name, tasks):
                try:
                    await asyncio.gather(*tasks)
                    print(f"Completed upload for {file_name}")
                except Exception as e:
                    print(f"Failed to upload {file_name}: {e}")
                    raise
            
            task = asyncio.create_task(upload_and_report(hdf5_file.name, upload_coroutines))
            background_tasks.append(task)
        
        # Wait for all background uploads to complete
        if background_tasks:
            print(f"\nWaiting for {len(background_tasks)} background uploads to complete...")
            await asyncio.gather(*background_tasks)
            print("All uploads completed!")
    
def main():
    """Main entry point for HDF5 uploader."""
    # Initialize uploader
    uploader = HDF5Upload()
    
    # Set directory first
    uploader.set_directory()
    
    # Verify files
    uploader.verifylocal()
    
    if not uploader.file_paths:
        print("No HDF5 files found to upload.")
        return
    
    # Run the async upload process
    asyncio.run(uploader.run())


if __name__ == "__main__":
    main()
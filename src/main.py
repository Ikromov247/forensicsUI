import sys
from pathlib import Path
import datetime

from .config import ApplicationConfig
from .application import ForensicsApplication
from .utils.input_handler import InputHandler
from .logging_config import setup_logging

def main():
    """
    Main entry point for the application.
    """
    try:
        # Setup logging
        log_dir = Path("logs")
        setup_logging(log_dir=log_dir)
        
        # Load configuration
        config_path = Path("config/default.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        config = ApplicationConfig.from_yaml(config_path)
        
        # Generate database name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.database.name = f"forensics_{timestamp}.db"
        
        # Initialize application
        app = ForensicsApplication(config)
        
        # Get input files
        input_handler = InputHandler()
        image_path, video_path = input_handler.get_input_files()
        
        # Process target image
        target_data = app.process_target_image(str(image_path))
        
        # Process video
        results = app.process_video(str(video_path), target_data)
        
        # Save results to database
        app.db_manager.save_results(results)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
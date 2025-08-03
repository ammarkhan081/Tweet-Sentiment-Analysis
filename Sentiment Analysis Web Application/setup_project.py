#!/usr/bin/env python3
"""
Setup script for Sentiment Analysis Flask App
Run this script to set up the complete project structure
"""

import os
import shutil

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'model',
        'static',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")

def move_files():
    """Move files to correct locations"""
    # Move CSS to static directory
    if os.path.exists('style.css'):
        shutil.move('style.css', 'static/style.css')
        print("✓ Moved style.css to static/style.css")
    
    # Move HTML templates to templates directory
    template_files = [
        'base.html',
        'index.html', 
        'predict.html',
        'performance.html',
        'results.html'
    ]
    
    for template in template_files:
        if os.path.exists(template):
            if template == 'predict.html':
                # Rename predict.html to predict_page.html
                shutil.move(template, 'templates/predict_page.html')
                print(f"✓ Moved {template} to templates/predict_page.html")
            else:
                shutil.move(template, f'templates/{template}')
                print(f"✓ Moved {template} to templates/{template}")

def main():
    print("Setting up Sentiment Analysis Flask App...")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Move files to correct locations
    move_files()
    
    print("\n" + "=" * 50)
    print("Setup completed! Next steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Create demo models: python create_demo_models.py")
    print("3. Run the app: python app.py")
    print("4. Visit: http://localhost:5000")

if __name__ == "__main__":
    main()
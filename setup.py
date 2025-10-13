"""
Setup and Installation Script for Enhanced Traffic Signal Classifier
This script helps you set up the environment and test the installation
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt not found!")
            return False
        
        # Install packages
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully!")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def check_tensorflow():
    """Check TensorFlow installation and GPU support"""
    print("\n🧠 Checking TensorFlow installation...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} installed")
        
        # Check GPU support
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU support available! Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("💻 No GPU found, will use CPU (training will be slower)")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow not installed or not working")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {e}")
        return False

def check_other_dependencies():
    """Check other important dependencies"""
    print("\n🔍 Checking other dependencies...")
    
    dependencies = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib.pyplot',
        'PIL': 'PIL.Image',
        'tkinter': 'tkinter',
        'cv2': 'cv2'
    }
    
    results = {}
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✅ {name} - OK")
            results[name] = True
        except ImportError:
            print(f"❌ {name} - Missing")
            results[name] = False
        except Exception as e:
            print(f"⚠️  {name} - Warning: {e}")
            results[name] = False
    
    return all(results.values())

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = ['models', 'logs', 'results', 'temp']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created/verified: {directory}/")
        except Exception as e:
            print(f"❌ Error creating {directory}: {e}")
            return False
    
    return True

def test_basic_functionality():
    """Test if the main components work"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports
        from main import TrafficSignalClassifier
        print("✅ Main classifier import - OK")
        
        from utils import ImageProcessor, ModelEvaluator
        print("✅ Utility functions import - OK")
        
        # Test classifier creation
        classifier = TrafficSignalClassifier()
        print("✅ Classifier instantiation - OK")
        
        # Test model creation (small model for testing)
        model = classifier.create_enhanced_model(num_classes=3)
        print("✅ Model creation - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def show_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    print("=" * 30)
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Memory info (basic)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM: {memory.total // (1024**3)} GB ({memory.available // (1024**3)} GB available)")
    except ImportError:
        print("RAM: Information unavailable (install psutil for details)")

def show_next_steps():
    """Show next steps for the user"""
    print("\n🚀 NEXT STEPS:")
    print("=" * 20)
    print("1. Run the application:")
    print("   python main.py")
    print("")
    print("2. Choose option 2 for GUI interface")
    print("")
    print("3. For training a new model:")
    print("   • Prepare your dataset in the required structure")
    print("   • Use option 1 for command line training")
    print("   • Or use 'Train New Model' button in GUI")
    print("")
    print("4. For testing with existing model:")
    print("   • Use 'Load Model' in GUI")
    print("   • Select images for classification")
    print("")
    print("5. Check README.md for detailed usage instructions")

def main():
    """Main setup function"""
    print("🚦 Enhanced Traffic Signal Classifier - Setup")
    print("=" * 50)
    
    # System info
    show_system_info()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install requirements")
        sys.exit(1)
    
    # Check TensorFlow
    if not check_tensorflow():
        print("\n❌ Setup failed: TensorFlow issues")
        sys.exit(1)
    
    # Check other dependencies
    if not check_other_dependencies():
        print("\n⚠️  Warning: Some dependencies missing, but continuing...")
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed: Could not create directories")
        sys.exit(1)
    
    # Test functionality
    if not test_basic_functionality():
        print("\n❌ Setup failed: Basic functionality test failed")
        sys.exit(1)
    
    # Success!
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 35)
    print("Your Enhanced Traffic Signal Classifier is ready to use!")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
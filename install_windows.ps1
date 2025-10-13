# Enhanced Traffic Signal Classifier - Windows Installation Script
# Run this script in PowerShell to set up the environment

Write-Host "🚦 Enhanced Traffic Signal Classifier - Windows Setup" -ForegroundColor Cyan
Write-Host "=" * 55 -ForegroundColor Cyan

# Check Python installation
Write-Host "`n🐍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.([8-9]|\d{2})") {
        Write-Host "✅ $pythonVersion - Compatible!" -ForegroundColor Green
    } else {
        Write-Host "❌ $pythonVersion - Requires Python 3.8 or higher!" -ForegroundColor Red
        Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Python not found! Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "`n📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "`n📚 Installing required packages..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ All packages installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Package installation failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ requirements.txt not found!" -ForegroundColor Red
    Write-Host "Installing core packages manually..." -ForegroundColor Yellow
    
    $packages = @(
        "tensorflow>=2.10.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "seaborn>=0.11.0"
    )
    
    foreach ($package in $packages) {
        Write-Host "Installing $package..." -ForegroundColor Cyan
        python -m pip install $package
    }
}

# Create necessary directories
Write-Host "`n📁 Creating project directories..." -ForegroundColor Yellow
$directories = @("models", "logs", "results", "temp")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "✅ Created: $dir/" -ForegroundColor Green
    } else {
        Write-Host "✅ Verified: $dir/" -ForegroundColor Green
    }
}

# Test TensorFlow
Write-Host "`n🧠 Testing TensorFlow installation..." -ForegroundColor Yellow
$testTF = python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, 'installed successfully'); gpus=tf.config.list_physical_devices('GPU'); print('GPUs found:', len(gpus))" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host $testTF -ForegroundColor Green
} else {
    Write-Host "⚠️ TensorFlow test failed, but continuing..." -ForegroundColor Yellow
}

Write-Host "`n🎉 SETUP COMPLETED!" -ForegroundColor Green
Write-Host "=" * 25 -ForegroundColor Green

Write-Host "`n🚀 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Run the application: python main.py" -ForegroundColor White
Write-Host "2. Choose option 2 for GUI interface" -ForegroundColor White
Write-Host "3. For help: Check README.md" -ForegroundColor White

Write-Host "`nPress any key to launch the application..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Launch the application
Write-Host "`n🚀 Launching Enhanced Traffic Signal Classifier..." -ForegroundColor Cyan
python main.py
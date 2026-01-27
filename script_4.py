import os

# Create directory structure first
directories = [
    'services/video-ingestion',
    'services/ai-inference', 
    'services/computer-vision',
    'services/threat-intelligence',
    'services/configuration',
    'services/api-gateway',
    'services/event-processing',
    'services/vector-search',
    'frontend/dashboard',
    'models',
    'config',
    'monitoring',
    'database/init'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    
print("✅ Directory structure created!")
for dir in directories:
    print(f"   📁 {dir}/")
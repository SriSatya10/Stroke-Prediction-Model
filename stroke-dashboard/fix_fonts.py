import glob, re

files = glob.glob('src/**/*.jsx', recursive=True)
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Strip the inline assignments
    content = content.replace(" fontFamily: 'DM Mono',", "")
    content = content.replace(", fontFamily: 'DM Mono'", "")
    content = content.replace(" fontFamily: 'Fraunces',", "")
    content = content.replace(", fontFamily: 'Fraunces'", "")
    
    content = content.replace(' fontFamily="DM Mono"', '')
    content = content.replace("fontFamily: 'DM Mono', ", "")
    content = content.replace("fontFamily: 'Fraunces', ", "")
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

print('Done')

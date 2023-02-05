python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

gdown --fuzzy 'https://drive.google.com/file/d/1oNs3dNAq0r0yUInF9oppiJF7kp0pKt63/view?usp=sharing'
unzip cleaned_files_civile.zip -d ./dataset
rm cleaned_files_civile.zip

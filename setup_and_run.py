import os

def clone_and_run():
    # Clone your GitHub repository
    repository_url = "https://github.com/dopamine333/handwritten_digits_recognition.git"  # 替換成你的 GitHub Repositories URL
    os.system(f"git clone {repository_url}")

    # Move into the cloned repository directory
    repository_name = "handwritten_digits_recognition"  # 替換成你的 Repositories 名稱
    os.chdir(repository_name)

    # Install dependencies
    os.system("pip install -r requirements.txt")

    # Run the main Python script
    os.system("python handwritten_digits_recognition_show_probability.py")

if __name__ == "__main__":
    clone_and_run()

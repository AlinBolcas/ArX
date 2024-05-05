git status
git add .
git rm -f --cached src/local_models/shap-e
git remote add origin git@github.com:AlinBolcas/ARV-0.git
git commit -m "Your commit message"
git push origin main


Certainly! Here's a comprehensive guide to frequently used Git commands and operations that will help you manage your GitHub repositories effectively. This guide covers initialization, branching, committing changes, working with remotes, handling SSH keys, and more. You can save these commands for future reference.

### Setting Up and Initializing Repositories

- **Initialize a New Repository**
  ```bash
  git init
  ```

- **Clone an Existing Repository**
  ```bash
  git clone https://github.com/username/repository.git
  ```

- **Add a Remote Repository**
  ```bash
  git remote add origin https://github.com/username/repository.git
  ```

- **View Configured Remotes**
  ```bash
  git remote -v
  ```

### Basic Operations: Add, Commit, Push, and Pull

- **Add Changes to Staging Area**
  ```bash
  git add .
  # Adds all changes in the working directory
  ```

- **Commit Changes**
  ```bash
  git commit -m "Commit message"
  ```

- **Push Changes to Remote Repository**
  ```bash
  git push origin main
  # Replace 'main' with your branch name if different
  ```

- **Pull Latest Changes from Remote Repository**
  ```bash
  git pull origin main
  ```

### Branching and Merging

- **Create a New Branch**
  ```bash
  git branch branch-name
  ```

- **Switch to a Branch**
  ```bash
  git checkout branch-name
  ```

- **Merge a Branch**
  ```bash
  git merge branch-name
  # Merge 'branch-name' into the current branch
  ```

### Handling SSH Keys for Authentication

- **Generate a New SSH Key Pair**
  ```bash
  ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
  ```

- **Add SSH Key to ssh-agent**
  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_rsa
  ```

- **Add SSH Key to GitHub Account**
  - Copy your SSH public key to the clipboard:
    ```bash
    pbcopy < ~/.ssh/id_rsa.pub
    # On macOS. Use `clip` on Windows, or `xclip` on Linux.
    ```
  - Go to GitHub > Settings > SSH and GPG keys > New SSH key, paste your key, and save.

### Other Useful Commands

- **Check Status of Your Working Directory**
  ```bash
  git status
  ```

- **View Commit History**
  ```bash
  git log
  ```

- **Create a New Repository on GitHub (via GitHub CLI)**
  ```bash
  gh repo create repository-name
  ```

- **Force Push (use with caution)**
  ```bash
  git push --force origin main
  ```

- **Configure Git to Use a Specific Branch as Default**
  ```bash
  git branch -m old-default new-default
  git fetch origin
  git branch -u origin/new-default new-default
  git remote set-head origin -a
  ```

- **Stashing Changes**
  - Stash changes in a dirty working directory:
    ```bash
    git stash
    ```
  - Apply stashed changes back:
    ```bash
    git stash pop
    ```

This collection of commands should serve as a solid foundation for most of your Git and GitHub operations. Remember, the specific commands you use can vary based on your workflow, the configuration of your project, and personal or team preferences. Always ensure you understand the implications of commands like `git push --force` to avoid unintended data loss.
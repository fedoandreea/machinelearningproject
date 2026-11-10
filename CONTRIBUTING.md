# How we work together

The idea: **`main` always works.** Nobody commits to `main` directly. All work happens on a **branch** and comes into `main` through a **pull request (PR)** that a teammate has looked at.

```
main ──●───────────────●──────────●──   (always working)
        \             /          /
         ●──●──●──●──┘  PR #1   /       ofri/task1-fix
              \                /
               ●──●──●────────┘  PR #2  andreea/m1-tree
```

## 1. Tasks: GitHub Issues
- Each piece of work is an **Issue**: go to *Issues → New issue → Task*.
- **Assign yourself** to the issue you are working on, so we never do the same thing twice.
- Optional: use *Projects → New project → Board* with columns **To do / In progress / In review / Done** to see everything at a glance.

## 2. The everyday workflow (copy these commands)

```bash
# 1. Start from an up-to-date main
git switch main
git pull

# 2. Create your own branch:  <your-name>/<short-topic>
git switch -c ofri/task1-fix

# 3. Work... then save your progress (commit often, small steps)
git status                      # see what changed
git add notebooks/task1_linear_regression.ipynb
git commit -m "Task 1: fix swapped train/test files"

# 4. Upload your branch to GitHub
git push -u origin ofri/task1-fix     # first time; afterwards just: git push
```

5. **Open a pull request.** On GitHub, click the yellow **"Compare & pull request"** banner and fill in the template. Write `Closes #<issue number>` so the issue closes automatically when the PR is merged.
6. **Review.** At least **one teammate** reads the PR. They can comment or approve it.
7. **Merge.** Click **"Squash and merge"**, then **"Delete branch"**.
8. **Back to step 1** for the next task. Run `git switch main && git pull` first.

### Keeping your branch up to date
If `main` changed while you were working:
```bash
git switch main && git pull
git switch ofri/task1-fix
git merge main
```

## 3. Rules for notebooks (important!)
Notebooks are big JSON files, and **two people editing the same notebook = merge conflict nightmare**.
- **Every notebook has one owner** (see the table in the README). Only the owner edits it. Everyone else comments in the PR.
- Code that several notebooks need (loading data, feature engineering, our from-scratch models) goes in **`src/`** as `.py` files, which merge cleanly.
- Before committing a notebook, run **Kernel → Restart & Run All**, so it runs top to bottom without errors.
- Use **relative paths** (`../data/case1/train1.csv`), never `C:\Users\...`, so the code works on everyone's computer.
- Save figures for the report to `report/figures/` with `plt.savefig("../report/figures/<name>.png", dpi=200, bbox_inches="tight")`.

## 4. Commit messages
Short and specific: say *what* changed.
- ✅ `Task 2: add from-scratch decision tree split function`
- ✅ `EDA: correlation heatmap for Case 2`
- ❌ `update`, `stuff`, `final2`

## 5. ML rules for the whole group
- **The test set is used once**, at the very end. All model and hyperparameter choices are made with cross-validation on train.
- **Case 2: split by pump** (`GroupKFold` with `groups=pump_id`), because the test pumps are different pumps.
- Fit scalers and PCA **inside** a `Pipeline`, so they only ever see training data.
- Fix random seeds (`random_state=42`) so results are reproducible.
- Write down any GenAI use. The report must cite it.

## 6. One-time setup (repo owner)
To make the "no direct commits to main" rule automatic:
1. **Add collaborators:** *Settings → Collaborators → Add people* (every group member).
2. **Protect main:** *Settings → Branches → Add branch protection rule* (or *Rules → Rulesets*). Use the branch name pattern `main` and tick **Require a pull request before merging** and **Require approvals: 1**.

## 7. Help, something went wrong
| Problem | Fix |
|---|---|
| `git push` rejected ("fetch first") | `git pull`, then `git push` again |
| Merge conflict in a `.py` file | Open the file, keep the right lines between `<<<<<<<` and `>>>>>>>`, then `git add` the file and `git commit` |
| Merge conflict in a notebook | Keep one version: `git checkout --theirs <nb>` (theirs) or `--ours` (yours), then re-apply your change by hand |
| Committed to `main` by mistake (not pushed yet) | `git switch -c my-branch` (your commit moves with you), then `git switch main && git reset --hard origin/main` |

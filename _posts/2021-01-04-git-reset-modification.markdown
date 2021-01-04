---
layout: post
title:  "git command to recover the modification"
excerpt: "git command without using git add, with using git add and git commit"
date:   2021-01-04 20:00:00
mathjax: true
---

### without using git add

for single revised file:

```sh
git checkout -- filename
```

for all the revised files:

```sh
git checkout .
```

### with using git add

for single added file:

```sh
git reset HEAD filename
```

for all the added files:

```sh
git reset HEAD 
```

### with using git commit

return to the last commit state:

```sh
git reset --hard HEAD^
```

return to the specific commit state:

```sh
# git log: query the commit ids
git reset --hard commit_id
```
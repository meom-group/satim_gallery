
## To see the published website, it's there: [https://meom-group.github.io/satim_gallery](https://meom-group.github.io/satim_gallery)

## Modifications en local

La première fois, il faut d'abord "cloner" le répertoire, c'est-à-dire le télécharger en local :

```bash
git clone https://github.com/meom-group/meom-group.github.io.git
```

Un répertoire meom-group.github.io est ainsi créé.
On peut alors modifier les fichiers ou en ajouter.
Pour transmettre les modifs effectuées, 3 lignes de code sont nécessaires :


```bash
git add .
git commit -m "commentaire à joindre aux modifs"
git push
```

Avant de faire les prochaines modifs, pour prendre en compte celles qui on été faites entre-temps par d'autres personnes, une mise à jour est nécessaire :


```bash
git pull
```

Si vous voulez prévisualiser l'effet que vos modifications auront sur le site avant de les "commiter", il suffit d'utiliser l'outil [jekyll](https://jekyllrb.com/) et de lancer la commande jekyll-serve. Vous pourrez alors voir le site web en local.


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/meom-group/satim_gallery/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/meom-group/satim_gallery/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

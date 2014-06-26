.SUFFIXES: .md .html .pdf .tex
.PRECIOUS: %.pdf %.html %.tex
.PRECIOUS: 

all: update-201406.pdf

# http://gregorkopf.de/blog/posts/impress.html

# impress.js
#.md.html: 
#	pandoc --toc --mathjax=mathjax/MathJax.js \
#	   --template=impress-template.html \
#	    -V impress-url=impress.js -s -t html5 \
#	    --highlight-style pygments \
#	    -V "subtitle:PhD Proposal" \
#	    -f markdown --section-divs -o $@ $<
.md.html: 
	pandoc --toc  \
	    --data-dir=pandoc-data \
	    --template=reveal.js.html.template \
	    -s -t html5 \
	    --highlight-style=pygments \
	    --section-divs -o $@ $<

.md.pdf:
	pandoc --listings -H header.tex \
	    --data-dir=pandoc-data \
	    --highlight-style=pygments \
	    -V theme:Warsaw -V colortheme:seahorse \
	    -s -t beamer -f markdown -o $@ $<
#	    --latex-engine=xelatex \

.md.tex:
	pandoc --listings -H header.tex \
	    --data-dir=pandoc-data \
	    --highlight-style=pygments \
	    -V theme:Warsaw -V colortheme:seahorse \
	    -s -t beamer -f markdown -o $@ $<
#	    --latex-engine=xelatex \

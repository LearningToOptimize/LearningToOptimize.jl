# Builds the diagrams of this directory: PDF with pdflatex, SVG with pdftocairo,
# then a white rectangle under the drawing so it reads on dark pages too.
# Each source gives a wide layout (<name>.svg) and a narrow one
# (<name>-compact.svg), selected by the job name.
#
#   make -C docs/src/assets/diagrams/src
#
# Needs a TeX Live with pdflatex and poppler's pdftocairo.
OUT := ..
TMP := build

# <name>:<source .tex without extension>
FIGS := learningtooptimize--method:method l2osobolev--regularizer:sobolev \
        learningtooptimize--ecosystem:ecosystem learningtooptimize--types:types
NAMES := $(foreach f,$(FIGS),$(word 1,$(subst :, ,$(f))))
base = $(patsubst %-compact,%,$(1))
src = $(word 2,$(subst :, ,$(filter $(call base,$(1)):%,$(FIGS))))

all: $(foreach n,$(NAMES),$(OUT)/$(n).svg $(OUT)/$(n)-compact.svg)

.SECONDEXPANSION:
$(TMP)/%.pdf: $$(call src,$$*).tex l2ostyle.tex
	mkdir -p $(TMP)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(TMP) -jobname=$* $< >/dev/null

$(OUT)/%.svg: $(TMP)/%.pdf
	pdftocairo -svg $< $(TMP)/$*.raw.svg
	perl -0pe 's/(<svg [^>]*>)/$$1\n<rect width="100%" height="100%" fill="#ffffff"\/>/' $(TMP)/$*.raw.svg > $@

clean:
	rm -r -f build

.PHONY: all clean

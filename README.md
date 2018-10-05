# OCR medical insurance document classifier

This repo contains a scanned Document/Text classification solution for an insurance industry client which focussed on grouping medical/health insurance claims into pre-defined categories. 

I developed a Deep Learning based framework which ensembled learnings from Document layout and structure, the content/text and amalgamation of consistent & coherent expert opinions. The framework helped in automation of the existing process leading to better efficiency and efficacy. I had a set of 40k scanned images of medical insurance documents and built an algorithm to classify those documents into given 5 categories. These scanned documents exhibited characteristics for each of the classes based on the document structure and token sequences present in the document.

The approach to classification is two-fold:
1. Train a CNN to learn the spatial features of the documents (logos, form structures etc..)
2. Run OCR on the documents to extract the textual content from the scanned documents
3. Train an LSTM classifier to learn the tokenization within each category of the documents
4. Ensemble the CNN and the LSTM classification scores to predict the class of the scanned document

The codes in this repo were tested on sample data.

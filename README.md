# 🌟 AI Engineering Project 🌟

Welcome to my repository! This collection features some of the most cutting-edge and innovative projects in the realm of AI, showcasing the immense potential and versatility of modern artificial intelligence technologies. Dive in and explore how these projects leverage advanced algorithms, powerful frameworks, and intricate mathematical insights to create solutions that push the boundaries of what's possible.

## 📚 Project 1: OpenAI LangChain GoogleAI Chatbot 🤖

### 🌐 Overview
Embark on an exploration of conversational AI with my **OpenAI LangChain GoogleAI Chatbot**. This project integrates the capabilities of OpenAI's language models with LangChain and GoogleAI to deliver seamless, context-aware conversations.

### 🚀 Features
- **Natural Language Understanding**: Harness the power of state-of-the-art NLP techniques.
- **Contextual Awareness**: Maintain coherent and relevant dialogues across multiple interactions.
- **Easy Integration**: Plug and play with various applications and platforms.

### 🔧 Key Technologies
- **OpenAI**: Elevating language comprehension and generation.
- **LangChain**: Chaining prompts for enhanced dialogue coherence.
- **GoogleAI**: Superior NLU & NLG for nuanced understanding.

### 🧠 Mathematical Insights
- **Transformers & Attention Mechanism**: The foundation of our language model in **NLP**, mathematically represented as:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  - **$Q$ (Query)**: The set of queries.
  - **$K$ (Key)**: The set of keys.
  - **$V$ (Value)**: The set of values.
  - **$$\(d_k\)$$**: The dimension of the key vectors.

  The attention mechanism computes a weighted sum of the values $\(V\)$, where the weights are determined by the compatibility of the queries $\(Q\)$ with the keys $\(K\)$.

## Cross-Entropy Loss
- **Cross-Entropy Loss**: Ensuring model accuracy by minimizing:
  $$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$
  - **$$\(y_i\)$$**: The true label (either 0 or 1).
  - **$$\(\hat{y}_i\)$$**: The predicted probability of the instance being in class 1.
  
  Cross-Entropy Loss measures the performance of a classification model whose output is a probability value between 0 and 1.

## Named Entity Recognition (NER) with Conditional Random Fields (CRFs)
### 📊 Results
Experience the magic of human-like conversations and see how our chatbot can transform user interactions.

## 📄 Project 2: Resume Parser with Named Entity Recognition 📑

### 🌐 Overview
Meet our **Resume Parser with Named Entity Recognition**, a tool designed to streamline the process of extracting vital information from resumes. Perfect for HR and recruitment automation, this parser accurately identifies names, contact details, skills, and more.

### 🚀 Features
- **Accurate Information Extraction**: Leveraging advanced NER techniques.
- **Data Structuring**: Organizes extracted data into structured formats for easy analysis.
- **Customizable Patterns**: Adapt the parser to specific needs using regex.

### 🔧 Key Technologies
- **spaCy**: Powerful NLP library for robust entity recognition.
- **pandas**: Efficient data manipulation and analysis.
- **re (regex)**: Custom pattern matching for tailored extraction.

### 🧠 Mathematical Insights
- **Named Entity Recognition (NER)**(Utilizing Conditional Random Fields (CRFs) for sequence labeling):
- NER is used to identify and classify entities in text into predefined categories.
  $$P(y \mid x) = \frac{1}{Z(x)} \exp \left( \sum_{i=1}^{n} \sum_{k} \lambda_k f_k(y_{i-1}, y_i, x, i) \right)$$
  - **$$\(P(y \mid x)\)$$**: The probability of label sequence \(y\) given observation sequence \(x\).
  - **$$\(Z(x)\)$$**: The normalization factor (partition function) ensuring the probabilities sum to 1.
  - **$$\(\lambda_k\)$$**: Parameters to be learned.
  - **$$\(f_k\)$$**: Feature functions.
   
- **Evaluation Metrics**: Precision, Recall, and F1-Score to measure performance:
  $$\text{Precision} = \frac{TP}{TP + FP}$$, $$\text{Recall} = \frac{TP}{TP + FN}$$, $$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  - **TP (True Positives)**: The number of correctly predicted positive instances.
  - **FP (False Positives)**: The number of incorrectly predicted positive instances.
  - Precision measures the accuracy of the positive predictions.
  - **FN (False Negatives)**: The number of positive instances that were incorrectly predicted as negative.
  - Recall measures the ability of the model to identify all relevant instances.
  - The F1-Score is the harmonic mean of precision and recall, providing a balance between the two metrics.
  - It is particularly useful when one need to take both false positives and false negatives into account.

### 📊 Results
Achieve high accuracy in information extraction and elevate your recruitment process to the next level.

## 📝 Project 3: Professional Profile Parser 🏆

### 🌐 Overview
The **Professional Profile Parser** is your go-to tool for extracting and analyzing key elements from professional profiles. From qualifications and experience to skills and achievements, this parser delivers comprehensive insights crucial for talent management.

### 🚀 Features
- **Comprehensive Parsing**: Extracts a wide range of profile elements.
- **Advanced NLP Techniques**: Ensures accurate and reliable data extraction.
- **Scalable**: Adaptable to various document formats and structures.

### 🔧 Key Technologies
- **NLTK**: Essential NLP library for text processing and analysis.
- **Gensim**: Sophisticated topic modeling and similarity detection.
- **sklearn**: Machine learning algorithms for profile analysis.

### 🧠 Mathematical Insights
- **Latent Dirichlet Allocation (LDA)**(For topic modeling, described by):-
  $$p(\theta, z, w \mid \alpha, \beta) = p(\theta \mid \alpha) \prod_{n=1}^{N} p(z_n \mid \theta) p(w_n \mid z_n, \beta)$$
  - **$$\mathcal{(\theta\)}$$**: Topic distribution for documents.
  - **$$\(z\)$$**: Topic assignment for each word.
  - **$$\(w\)$$**: Words in documents.
  - **$$\mathcal{(\alpha, \beta\)}$$**: Hyperparameters for the Dirichlet prior on the per-document topic distributions and per-topic word distribution respectively.
    
- **Cosine Similarity**(Measuring text similarity):-
- Cosine similarity is a measure used to calculate the similarity between two vectors.
- It is commonly used in text analysis and natural language processing to measure how similar
  two pieces of text are. The formula for cosine similarity between two vectors $$\(A\)$$ and $$\(B\)$$ is:
  $$\mathcal{cosine^-\delta} = \frac{A \cdot B}{\|A\| \|B\|}$$
- **Dot Product ( \( A \cdot B \) )**:
  $$A \cdot B = \mathcal{\sum_{i=1}^{n} A_i B_i}$$
  This represents the sum of the products of the corresponding entries of the two vectors.
- **Norm ( $$\( \|A\| \$$) )**:
  $$\|A\| = \mathcal{\sqrt{\sum_{i=1}^{n} A_i^2}}$$
  This represents the magnitude (length) of vector $$\(A\)$$.

### 📊 Results
Unlock deep insights into professional profiles and make informed talent management decisions.

---

### 🚀 How to Get Started
1. **Clone the Repository**: `git clone https://github.com/suvro5495/AI_Engineering_Projects.git`
2. **Explore the Projects**: Navigate through each project directory to discover detailed instructions and code.
3. **Run the Notebooks**: Open the Jupyter notebooks and see the magic unfold in real-time.

### 🌟 Contributing
I welcome contributions! Feel free to fork the repository, create a new branch, and submit a pull request with your enhancements and innovations.

### 📫 Contact
For any queries, reach out me at [suvro5495@gmail.com](suvro5495@gmail.com).

---

✨ Thank you for exploring my AI Engineering Project! ✨

Dive into the world of AI and witness the convergence of technology, machine learning mathematics, and innovation like never before. 🚀🔍📊

---

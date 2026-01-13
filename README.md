
---

# **Customer Support Automation — Transformer-Based NLP System**

Automated classification of customer support tickets using a fine-tuned **DistilBERT** model to route requests into operational queues (e.g., Technical Support, Billing, Product Support) with high accuracy and controllable misrouting risk.

This project demonstrates how **modern Transformer NLP** can replace manual ticket triage with scalable, data-driven automation.

---

## **Problem**

Customer support teams receive large volumes of unstructured text (subject + message body).
Manually reading and routing tickets is:

* Slow
* Error-prone
* Inconsistent across agents

Misrouted tickets increase resolution time and customer dissatisfaction.

The goal is to **automatically assign each ticket to the correct queue** using machine-learned intent detection.

---

## **Solution**

This project builds a **multi-class Transformer-based classifier** that:

* Reads raw ticket text
* Learns customer intent from historical data
* Predicts the correct support queue

The model is trained using **DistilBERT (English)**, fine-tuned on labeled customer support tickets.

The final model is saved as:

```
distilbert_queue_classifier.keras
```

and is ready for integration into a production support workflow.

---

## **Model Architecture**

* **Base model:** `distil_bert_base_en_uncased`
* **Framework:** TensorFlow + Keras NLP
* **Task:** Multi-class text classification
* **Output:** Softmax over support queues

DistilBERT provides:

* Transformer-level language understanding
* Lower latency than full BERT
* Production-friendly deployment size

---

## **ML Pipeline**

1. **Text ingestion**

   * Ticket subject
   * Ticket body

2. **Preprocessing**

   * Text normalization
   * Label consolidation
   * Class balancing to handle skewed queues

3. **Model training**

   * Fine-tuned DistilBERT
   * Adam optimizer (5e-5 learning rate)
   * Softmax classification head

4. **Evaluation**

   * Per-class:

     * True Positives
     * False Positives
     * False Negatives
     * True Negatives
   * Not just accuracy — misrouting risk is explicitly measured

5. **Model export**

   * Saved as `.keras` for downstream inference

---


## **Key Use Cases**

This system can be used for:

* Automated ticket routing
* Priority escalation detection
* Queue load balancing
* Customer experience optimization

---

## **Repository Structure**

```
Dataset_Training_Model_Training.ipynb   # End-to-end data processing, training, and evaluation
distilbert_queue_classifier.keras      # Trained Transformer model
```

---

## **How to Run**

1. Open the notebook:

```
Dataset_Training_Model_Training.ipynb
```

2. Install dependencies (TensorFlow, Keras NLP, Pandas, NumPy)

3. Run all cells to:

   * Load data
   * Train DistilBERT
   * Evaluate per-class performance
   * Save the trained model


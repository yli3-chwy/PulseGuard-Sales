# Anomaly Detection in Chewy's Daily Sales Data

- [Anomaly Detection in Chewy's Daily Sales Data](#anomaly-detection-in-chewys-daily-sales-data)
  - [Overview](#overview)
  - [Sync](#sync)
    - [1. Previous Brainstorm Insights](#1-previous-brainstorm-insights)
    - [2. "Product" / Model Components](#2-product--model-components)
    - [3. Challenges and Unclear Questions](#3-challenges-and-unclear-questions)
  - [Approach](#approach)
    - [1. Why LSTM](#1-why-lstm)
    - [2. Implementation](#2-implementation)
    - [3. Performance Metrics](#3-performance-metrics)
    - [4. Next Steps](#4-next-steps)
  - [Conclusion](#conclusion)





## Overview


In our pursuit to enhance anomaly detection in Chewy's daily sales data, I delved into various methods and compiled the findings in a document titled "Anomaly Detection 101." This report aims to provide a straightforward overview of our exploration and the chosen approach for anomaly detection.

## Sync

### 1. Previous Brainstorm Insights

- Focus on detecting anomaly behavior among all SKUs simultaneously.
- Detect changes in SKU behavior over time.
- Identify clusters of SKU-level anomalies and find shared causes.

### 2. "Product" / Model Components

- Component 1: Flag individual SKUs with anomalous behavior based on primary metrics.
- Component 2: Group SKUs with similar anomalies based on secondary metrics.
- Component 3: Identify additional features shared by SKUs in each anomaly group.

### 3. Challenges and Unclear Questions

- How do we "label" anomalies
- Metric selection in Component 2
- Addressing differences in error significance for diverse products.
- Accounting for baseline metric uncertainty and weighting different metric changes.

## Approach

Building on these insights, our selected approach involves leveraging Long Short-Term Memory (LSTM) networks for anomaly detection in Chewy's daily sales data.

### 1. Why LSTM

LSTM is like a smart tool for understanding sales data, helping a computer system remember important patterns and trends over both short and long periods, making it great for predicting and analyzing sales behaviors.

- LSTM excels in capturing temporal patterns, making it well-suited for time series data like daily sales.
- It accommodates the dynamic nature of sales metrics and provides robust anomaly detection.


### 2. Implementation

- Developed an LSTM model trained on sequences of sales, price, and inventory data.
- Calculated reconstruction errors to identify anomalies, allowing for precise detection.

### 3. Performance Metrics

Given the uncertainty in labeling "anomalies," common binary classification metrics such as precision, recall, and F1 score can be employed to evaluate the anomaly detection model's performance. The choice of these metrics allows for a comprehensive assessment, even in situations where the labeling process poses challenges.

### 4. Next Steps

- Integration with other methods, like Isolation Forest, for enhanced anomaly detection.
- Ongoing exploration of additional features and metrics for refining anomaly identification.


## Conclusion

Our chosen LSTM-based approach lays a strong foundation for effective anomaly detection in Chewy's dynamic sales environment. As we continue to refine and integrate methods, we aim to provide valuable insights into SKU-level anomalies and their shared causes.


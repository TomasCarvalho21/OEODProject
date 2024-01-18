# Master in Data Science - ISCTE: Reinforcement Learning for Algorithmic Trading

## Project Overview

This project focuses on developing and evaluating trading strategies for Micron Technology, Inc. (MU) using both traditional statistical methods and advanced reinforcement learning techniques. Our approach involves extracting and analyzing historical stock prices, employing statistical strategies based on Exponential Moving Averages (EMA), and designing a Deep Q-Learning model for algorithmic trading. We aim to fine-tune and compare these strategies to identify the most effective approach.

## Contributors
- Afonso Fareleiro
- Armando Ferreira
- Tomás Carvalho

## Group 12 - MU: Micron Technology, Inc

## Project Contents

1. **Data Extraction:** Historical stock prices of MU from 2019 to 2023 were obtained and analyzed.

2. **Statistical Analysis:** Returns, expected mean returns, risk metrics (volatility, Sharpe ratio, CAGR, Variance, CVAR) were calculated.

3. **EMA Trading Strategy:** A trading strategy based on short-term and long-term Exponential Moving Averages was developed.

4. **Strategy Evaluation:** The EMA strategy's performance was assessed using various financial metrics.

5. **Reinforcement Learning Model:** A Deep Q-Learning model was trained on MU's stock data for algorithmic trading.

6. **RL Strategy Evaluation:** The performance of the RL strategy was evaluated and compared to the EMA strategy.

7. **Algorithm Optimization:**
   - **EMA Optimization:** Parameters for the EMA strategy were fine-tuned.
   - **RL Hyperparameter Tuning:** The Deep Q-Learning model was optimized through hyperparameter tuning.

8. **Results Comparison and Interpretation:**
   - **EMA Strategies:** Non-optimized vs. optimized EMA strategies were compared.
   - **EMA vs. RL Strategy:** The optimized EMA strategy was compared with the optimized RL strategy.

## Conclusion

This project demonstrates the application of both traditional statistical methods and advanced machine learning techniques in the context of algorithmic trading. By comparing the performance of EMA-based strategies and a Reinforcement Learning model, we aim to uncover insights into the efficacy and applicability of these methods in real-world trading scenarios.

## GitHub Repository Structure

- `saved_models/`: Saved models of the Deep Q-Learning algorithm.
- `/main.ipynb`: Main jupyter notebook.
- `/ema.py`: Object-oriented implemenation of Exponential Moving Average strategy.
---

### How to Use This Repository

1. **Explore the Data:** Start by examining the historical stock data in the `data/` directory.

2. **Review Analysis:** Check the Jupyter notebooks in the `notebooks/` directory to understand the methodologies used.

3. **Experiment with Models:** Use the scripts in the `scripts/` directory to modify, train, and evaluate trading strategies.

4. **Analyze Results:** Dive into the `results/` directory for a comprehensive view of the strategies' performances.

---

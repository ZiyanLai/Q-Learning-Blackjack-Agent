# Q‑Learning for Blackjack

A reinforcement‑learning study comparing simple and full‑action agents against a rule‑based basic strategy.

---

## Overview

This project trains and evaluates Q‑learning agents to play Blackjack:

- **Hit‑or‑Stand Agent**: Learns with only two actions (hit, stand).  
- **All‑Actions Agent**: Learns with all five standard actions (hit, stand, double down, surrender, split).  
- **Basic Strategy**: A non‑learning, rule‑based benchmark.

Agents play millions of simulated hands and update a Q‑table to maximize long‑term rewards. We compare expected reward, win/tie/loss rates, and reward variance.

---

## Key Findings

- **Both Q‑learning agents outperform** the basic strategy in **expected reward**.  
- **Full‑action access** (split, double, surrender) **improves mean‑variance** performance, despite a slight drop in raw win rate.  
- The Hit‑or‑Stand agent already matches or exceeds basic strategy, showing the power of simple exploration/exploitation.

---

## Evaluation Metrics

- **Expected Reward** (mean payoff per hand)  
- **Win Rate / Tie Rate / Loss Rate**  
- **Reward Variance** (risk measure)

---

## Full Report

<p align="center">
  <object data="./Q-Learning for Blackjack.pdf" 
          type="application/pdf" 
          width="100%" height="600px">
    <p>Your browser does not support embedded PDFs.  
       You can <a href="./Q-Learning for Blackjack.pdf">download the full report here</a>.</p>
  </object>
</p>

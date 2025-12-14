# DSL Design Document

## 1. Overview
This project implements a Domain Specific Language (DSL) for algorithmic trading. It bridges the gap between natural language inputs and vectorised Python execution.

## 2. Grammar Specification
The DSL follows a standard `CONDITION` structure optimized for readability and parsing.

**Format:**
```text
STRATEGY ::= ACTION "WHEN" CONDITION
ACTION   ::= "ENTRY" | "EXIT"
CONDITION ::= EXPRESSION OPERATOR EXPRESSION [ "AND" | "OR" CONDITION ]

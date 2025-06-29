# RAFL-Fed: Resource-Aware Federated Learning for Citrate Anticoagulation Management using MIMIC-IV

## Overview

This repository introduces an innovative framework designed to **monitor and manage citric acid overdose** in clinical settings, adapting **regional citrate anticoagulation (RCA) protocols**. The proposed method leverages **Edge Computing** and **Federated Learning** to create a **privacy-preserving**, **resource-efficient**, and **adaptable system** for real-time healthcare applications.

## Key Features

- **Federated Learning**: Decentralized training that respects data privacy by keeping patient data on local devices.
- **Dynamic Client Selection**: Clients are selected based on available resources, ensuring robust and efficient model updates.
- **RAFL-Fed Algorithm**: Implements **Resource-Aware Federated Learning with Dynamic Client Selection**, which adaptively chooses participants and performs weighted model aggregation.
- **Real-World Clinical Data**: Uses the **MIMIC-IV dataset**, a comprehensive and publicly available electronic health record database.
- **Edge-Compatible**: Optimized for low-latency operation in hospital edge settings to enable timely updates and responses.
- **Clinical Impact**: Enables real-time adjustment of RCA protocols, improving **patient safety** and **treatment outcomes**.

## How It Works

1. **Client Initialization**: Each hospital or edge node (client) locally trains a model using its private data.
2. **Dynamic Client Selection**: The server evaluates client resources and selects a capable subset for the next round.
3. **Local & Auxiliary Updates**: Selected clients compute both local and auxiliary model updates.
4. **Model Aggregation**: The central server collects updates and performs **weighted averaging** to refine the global model.
5. **Communication Rounds**: The above cycle is repeated over multiple rounds, enabling continuous learning and adaptation to evolving data trends.

## Dataset

- **MIMIC-IV**: A freely accessible critical care dataset developed by MIT Lab for Computational Physiology.
- Link: [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)

> *Please ensure you have completed the required data usage agreements to access MIMIC-IV.*

## Clinical Relevance

This framework aims to revolutionize **bedside monitoring and adjustment** of citrate anticoagulation treatments. By decentralizing model training and enabling intelligent protocol adjustment in near-real-time, this approach has the potential to:

- Minimize risks of **citric acid overdose**
- Maximize **treatment efficacy**
- Enable **personalized and adaptive care** in critical environments



```bash
. ./venv/bin/activate
cd lightgbm

# level-wise vs leaf-wise
streamlit run demo_grow_strategy.py --server.port 8080

# gbdt vs goss vs efb
streamlit run demo_goss_vs_efb.py --server.port 8081
```

            st.info("Check that the model and dataset were created using the same feature columns.")

elif page == "Model Performance":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Evaluation and validation</div>
        <h1>Model Performance</h1>
        <p>Results from the final test set and cross-validation comparison of the trained classifiers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Random Forest — test set</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", "85.40%")
    m2.metric("Precision", "74.89%")
    m3.metric("Recall", "42.51%")
    m4.metric("F1-score", "54.23%")
    m5.metric("ROC-AUC", "85.24%")

    st.caption("The model was selected using the highest mean F1-score across stratified 5-fold cross-validation.")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="soft-card"><h3>Confusion matrix</h3><p class="muted">Actual vs. predicted labels on the 2,000-row test set.</p></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        matrix = [[1535, 58], [234, 173]]
        ax.imshow(matrix, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], labels=["Stayed (0)", "Churned (1)"])
        ax.set_yticks([0, 1], labels=["Stayed (0)", "Churned (1)"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown('<div class="soft-card"><h3>Cross-validation F1-score</h3><p class="muted">Mean F1-score for each candidate model.</p></div>', unsafe_allow_html=True)
        names = ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"]
        scores = [0.5404, 0.5085, 0.4989, 0.4867]
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        bars = ax.bar(names, scores)
        ax.set_ylim(0, 0.65)
        ax.set_ylabel("Mean F1-score")
        ax.set_title("Model Comparison")
        ax.tick_params(axis="x", rotation=18)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown('<div class="section-title">Cross-validation summary</div>', unsafe_allow_html=True)
    results = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"],
        "Accuracy": [0.8521, 0.7289, 0.8342, 0.7924],
        "Precision": [0.7363, 0.4032, 0.6491, 0.4904],
        "Recall": [0.4270, 0.6890, 0.4055, 0.4834],
        "F1-score": [0.5404, 0.5085, 0.4989, 0.4867],
        "ROC-AUC": [0.8440, 0.7804, 0.7749, 0.6774],
    })
    st.dataframe(
        results.style.format({col: "{:.2%}" for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}),
        use_container_width=True,
        hide_index=True,
    )
    st.warning("Recall for the churn class is 42.51%, so the model misses some customers who actually churn. This is an important limitation to discuss.")

else:
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Project documentation</div>
        <h1>About the Project</h1>
        <p>An end-to-end machine-learning project focused on identifying bank customers who may leave.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project workflow</div>', unsafe_allow_html=True)
    steps = st.columns(4)
    workflow = [
        ("01", "Data cleaning", "Handled missing values, duplicates, and data-quality issues."),
        ("02", "EDA & features", "Explored churn patterns and created useful features."),
        ("03", "Modeling", "Compared four classifiers using stratified cross-validation."),
        ("04", "Deployment", "Built an interactive Streamlit application for predictions."),
    ]
    for col, (number, title, description) in zip(steps, workflow):
        with col:
            st.markdown(
                f'<div class="soft-card"><div class="eyebrow">{number}</div><h3>{title}</h3><p class="muted">{description}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.markdown("""
    <div class="soft-card">
      <h3>🎯 Project objective</h3>
      <p class="muted">Predict whether a bank customer is likely to leave, helping inform customer-retention decisions.</p>
      <h3>🧰 Technologies</h3>
      <p class="muted">Python · Pandas · Scikit-learn · Joblib · Streamlit · Matplotlib</p>
      <h3>⚠️ Responsible use</h3>
      <p class="muted">The output is a statistical estimate and should be considered alongside human review and other relevant information.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#71839a;font-size:.8rem;padding:.4rem 0 1rem 0;">'
    'Bank Customer Churn Prediction · Machine Learning Project · Built with Streamlit</div>',
    unsafe_allow_html=True,
)

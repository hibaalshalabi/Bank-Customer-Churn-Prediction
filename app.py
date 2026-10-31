
                st.progress(churn_probability)
                st.caption("Probability estimated by the trained model.")

                st.markdown("</div>", unsafe_allow_html=True)

            st.caption(
                "This is a model prediction, not a guarantee "
                "of the customer's future behavior."
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")
            st.info(
                "Check that the model and dataset were created "
                "using the same feature columns."
            )


# ------------------------------------------------------------
# MODEL PERFORMANCE PAGE
# ------------------------------------------------------------
elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Evaluation and validation</div>
        <h1>Model Performance</h1>
        <p>
        Results from the final test set and cross-validation comparison
        of the trained classifiers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Random Forest — test set</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Accuracy", "85.40%")
    m2.metric("Precision", "74.89%")
    m3.metric("Recall", "42.51%")
    m4.metric("F1-score", "54.23%")
    m5.metric("ROC-AUC", "85.24%")

    st.caption(
        "The model was selected using the highest mean F1-score "
        "across stratified 5-fold cross-validation."
    )

    left, right = st.columns(2)

    # --------------------------------------------------------
    # CONFUSION MATRIX - BLUE COLOR GRADIENT
    # --------------------------------------------------------
    with left:

        st.markdown(
            '<div class="soft-card"><h3>Confusion matrix</h3>'
            '<p class="muted">Actual vs. predicted labels on the 2,000-row test set.</p></div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(5.2, 3.8))

        matrix = [[1535, 58], [234, 173]]

        image = ax.imshow(
            matrix,
            interpolation="nearest",
            cmap="Blues"
        )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")

        ax.set_xticks([0, 1], labels=["Stayed (0)", "Churned (1)"])
        ax.set_yticks([0, 1], labels=["Stayed (0)", "Churned (1)"])

        threshold = max(max(row) for row in matrix) / 2

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(matrix[i][j]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="white" if matrix[i][j] > threshold else "#102d52"
                )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --------------------------------------------------------
    # MODEL COMPARISON - READ FROM CSV FILE
    # --------------------------------------------------------
    with right:

        st.markdown(
            '<div class="soft-card"><h3>Cross-validation F1-score</h3>'
            '<p class="muted">Mean F1-score for each candidate model.</p></div>',
            unsafe_allow_html=True
        )

        if os.path.exists(RESULTS_PATH):

            try:
                results = pd.read_csv(RESULTS_PATH)

                required_columns = ["Model", "CV F1"]

                if all(column in results.columns for column in required_columns):

                    names = results["Model"]
                    scores = results["CV F1"]

                    fig, ax = plt.subplots(figsize=(5.2, 3.8))

                    bars = ax.bar(names, scores, color="#1769e0")

                    ax.set_ylim(0, 0.65)
                    ax.set_ylabel("Mean F1-score")
                    ax.set_title("Model Comparison")
                    ax.tick_params(axis="x", rotation=18)
                    ax.bar_label(bars, fmt="%.3f", padding=3)

                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                else:
                    st.warning(
                        "The comparison CSV must contain Model and CV F1 columns."
                    )

            except Exception as error:
                st.error(f"Could not read model comparison file: {error}")

        else:
            st.warning(
                "Model comparison file not found. "
                "Please add model_comparison_results.csv to the project folder."
            )

    st.markdown(
        '<div class="section-title">Model Limitations & Challenges (Demo Day Review)</div>',
        unsafe_allow_html=True
    )

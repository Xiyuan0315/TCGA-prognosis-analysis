import streamlit as st
import numpy as np
from base_dataset import CancerDataset
import metric
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

def get_model(model,weights):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=20, class_weight = weights),
        # 'rf_oob': RandomForestClassifier(n_estimators= 100, 
        #     oob_score= True, class_weight= weights),
        "Support Vector Machine": svm.SVC(kernel='poly',class_weight = weights),
        'Guassian Naive Bayes': GaussianNB(),
        "XGBoost": xgb.XGBClassifier(objective="binary:logistic", random_state=42,class_weight = weights)
    }
    return models.get(model)


def show_predict_page():
    st.title("Predict SKCM Prognosis")
    count_matrices = ('tpm',
    'tumor',
    'Tcell',
    'CD8Tex')

    genes = ('Correspond','ALL')

    models =("Random Forest",
        "Support Vector Machine",
        "Guassian Naive Bayes",
        "XGBoost")
    col1, col2 = st.columns(2)
    with col1:
        count_matrix = st.selectbox("Database(TPM or cell specific counting matrix)",count_matrices)
    with col2:
        gene = st.selectbox("Gene List(Featured genes from Cox regression)",genes )
        if gene == 'Correspond':
            gene = count_matrix # str


    model = st.selectbox("Model", models)
    ok = st.button("Predict Performance")
    if ok:
        skcm = CancerDataset(file_name = f'data/skcm_{count_matrix}.csv',matrix_type=gene)

        X_train, X_test, y_train, y_test = train_test_split(skcm.data, 
                                                        skcm.label, test_size=0.3,random_state=109)
        np.random.seed(315728)
        weights = class_weight.compute_class_weight('balanced', 
        classes=np.unique(y_train), y = y_train)
        clf = get_model(model,weights = dict(enumerate(weights)))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = metric.acc(y_test, y_pred)
        sen_spec,_ = metric.sens_spec(y_test, y_pred)
        sen,spec = round(sen_spec[0],3)*100,round(sen_spec[1],3)*100
        _,_,_,roc = metric.roc_auc(y_test,y_pred)
        feature_rank,feature_rank_plot = metric.get_feature_rank(clf, X_test, y_test,skcm.feature)

        st.subheader(f"Accuracy of {model} is {round(acc,3)*100}%.")
        st.write(f"With Sensitivity: {sen}% Vs. Specificity: {spec}%")
        st.write("""### ROC""")
        st.pyplot(roc)
        st.subheader("Feature Ranking")
        st.pyplot(feature_rank_plot)

        top_features, paras = st.columns(2)
        with top_features:
            with st.expander("Top 10 features"):
                st.write(feature_rank[0:10].index.tolist())
        with paras:
            with st.expander("Module parameters"):
                st.write(clf.get_params())
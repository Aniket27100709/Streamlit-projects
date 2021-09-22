import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

import joblib


pipe=joblib.load(open('emotion_classifier.pkl','rb'))

def pred_emot(words):
    result=pipe.predict([words])
    return result[0]
def predict_prob(words):
    result=pipe.predict_proba([words])
    return result
emoji={"anger":"😡","disgust":"😫","shame":"😣","surprise":"😱","fear":"🥶","sadness":"😥","joy":"😊","neutral":"😐"}
def main():
    st.title("Emotion Classifier")
    menu=["Home","Monitor",'About']
    choice=st.sidebar.selectbox("Menu",menu)

    if choice=="Home":
        st.subheader("Home emoticon")
        with st.form(key="emotion"):
            text=st.text_area("Text Here")
            submit=st.form_submit_button(label="Submit")
        if submit:
            c1,c2=st.beta_columns(2)
            pred=pred_emot(text)
            prob=predict_prob(text)
            with c1:
                st.success("Orignal words")
                st.write(text)
                st.success("Predicted")
                emojis=emoji[pred]
                st.write("{}:{}".format(pred,emojis))
                st.write("Confidence:{}".format(np.max(prob)))
            with c2:
                st.success("Probability of prediction")
                st.write(prob)
                df=pd.DataFrame(prob,columns=pipe.classes_)
                st.write(df.T)
                df_proper=df.T.reset_index()
                df_proper.columns=["emotions",'probability']

                fig=alt.Chart(df_proper).mark_bar().encode(x="emotions",y="probability",color="emotions")
                st.altair_chart(fig,use_container_width=True)
    elif choice=="Monitor":
        st.subheader("Monitoring")
    
    else:
        st.subheader("About")

    
if __name__=='__main__':
    main()
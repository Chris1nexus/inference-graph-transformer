import streamlit as st
import openslide
import os
from streamlit_option_menu import option_menu
from predict import Predictor



st.set_page_config(page_title="",layout='wide')
predictor = Predictor()





ABOUT_TEXT = "🤗 LastMinute Medical - Web diagnosis tool."
CONTACT_TEXT = """
_Built by Christian Cancedda and LabLab lads with love_ ❤️ 
[![Follow](https://img.shields.io/github/followers/Chris1nexus?style=social)](https://github.com/Chris1nexus)
[![Follow](https://img.shields.io/twitter/follow/chris_cancedda?style=social)](https://twitter.com/intent/follow?screen_name=chris_cancedda)
"""
VISUALIZE_TEXT = "Visualize WSI slide by uploading it on the provided window"
DETECT_TEXT = "Generate a preliminary diagnosis about the presence of  pulmonary disease"



with st.sidebar:
    choice = option_menu("LastMinute - Diagnosis",
                         ["About", "Visualize WSI slide", "Cancer Detection", "Contact"],
                         icons=['house', 'upload', 'activity',  'person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                             # "container": {"padding": "5!important", "background-color": "#fafafa", },
                             "container": {"border-radius": ".0rem"},
                             # "icon": {"color": "orange", "font-size": "25px"},
                             # "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                             #              "--hover-color": "#eee"},
                             # "nav-link-selected": {"background-color": "#02ab21"},
                         }
                         )
st.sidebar.markdown(
    """
<style>
.aligncenter {
    text-align: center;
}
</style>
<p class="aligncenter">
    <a href="https://twitter.com/chris_cancedda" target="_blank"> 
        <img src="https://img.shields.io/twitter/follow/chris_cancedda?style=social"/>
    </a>
</p>
    """,
    unsafe_allow_html=True,
)



if choice == "About":
    st.title(choice)



if choice == "Visualize WSI slide":
    st.title(choice)
    st.markdown(VISUALIZE_TEXT)

    uploaded_file = st.file_uploader("Choose a WSI slide file to diagnose (.svs)")
    if uploaded_file is not None:
        ori = openslide.OpenSlide(uploaded_file.name)
        width, height = ori.dimensions

        REDUCTION_FACTOR = 20
        w, h = int(width/512), int(height/512)
        w_r, h_r = int(width/20), int(height/20)
        resized_img = ori.get_thumbnail((w_r,h_r))
        resized_img = resized_img.resize((w_r,h_r))
        ratio_w, ratio_h = width/resized_img.width, height/resized_img.height
        #print('ratios ', ratio_w, ratio_h)
        w_s, h_s = float(512/REDUCTION_FACTOR), float(512/REDUCTION_FACTOR)   
        st.image(resized_img, use_column_width='never')   

if choice == "Cancer Detection":
    state = dict()
    
    st.title(choice)
    st.markdown(DETECT_TEXT)
    uploaded_file = st.file_uploader("Choose a WSI slide file to diagnose (.svs)")
    if uploaded_file is not None:
        # To read file as bytes:
        #print(uploaded_file)
        with open(os.path.join(uploaded_file.name),"wb") as f:
             f.write(uploaded_file.getbuffer())
        with st.spinner(text="Computation is running"):
            predicted_class, viz_dict = predictor.predict(uploaded_file.name)
        st.info('Computation completed.')
        st.header(f'Predicted to be: {predicted_class}')
        st.text('Heatmap of the areas that show markers correlated with the disease.\nIncreasing red tones represent higher likelihood that the area is affected')
        state['cur'] = predicted_class
        mapper = {'ORI': predicted_class, predicted_class:'ORI'}
        readable_mapper = {'ORI': 'Original',  predicted_class :'Disease heatmap' }        
        #def fn():
        #    st.image(viz_dict[mapper[state['cur']]], use_column_width='never', channels='BGR') 
        #    state['cur'] = mapper[state['cur']]
        #    return 

        #st.button(f'See {readable_mapper[mapper[state["cur"]] ]}', on_click=fn   )
        #st.image(viz_dict[state['cur']], use_column_width='never', channels='BGR') 
        st.image([viz_dict[state['cur']],viz_dict['ORI']], caption=['Original', f'{predicted_class} heatmap'] ,channels='BGR'
            # use_column_width='never', 
            ) 
            

if choice == "Contact":
    st.title(choice)
    st.markdown(CONTACT_TEXT)
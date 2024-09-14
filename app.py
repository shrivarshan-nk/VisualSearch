import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load pretrained model
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

# Function to preprocess images
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size as needed
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to get predictions
def predict_image(model, image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return np.argmax(predictions, axis=1)[0]  # Assuming the model returns class probabilities

# Load pretrained model (replace 'path/to/your/product_classifier.h5' with your actual model path)
model = load_model('product_classifier.h5')

# Sample data for products
base_path = r'Products'
products = [
    {'id': 1, 'name': 'Alphenlibe', 'price': 50, 'image': os.path.join(base_path, 'Alphenlibe', 'alphenlibe_icon.webp')},
    {'id': 2, 'name': 'Mariegold', 'price': 20, 'image': os.path.join(base_path, 'Mariegold', 'mariegold_icon.png')},
    {'id': 3, 'name': 'Mentos', 'price': 10, 'image': os.path.join(base_path, 'Mentos', 'mentos_icon.webp')},
    {'id': 4, 'name': 'Lays', 'price': 20, 'image': os.path.join(base_path, 'Lays', 'lays_icon.webp')},
    {'id': 5, 'name': 'Milkbikis', 'price': 20, 'image': os.path.join(base_path, 'Milkbikis', 'milkbikis_icon.png')}
]

# Convert the list to a DataFrame
df = pd.DataFrame(products)

# Initialize session state variables for cart and visual search
if 'cart' not in st.session_state:
    st.session_state.cart = {}

if 'visual_search' not in st.session_state:
    st.session_state.visual_search = False

st.title("E-Commerce Store")
# Use a container to control the layout
with st.container():
    # Add search bar and camera icon button with coordinates
    col1, col2 = st.columns([10, 1])
    
    with col1:
        search_query = st.text_input("Search for products","")
        
    with col2:
        st.write("\n")
        st.write("\n")
        if st.button("📷"):  # This adds a camera icon as the button
            st.session_state.visual_search = True
st.markdown("---")

# Visual search logic
if st.session_state.visual_search:
    uploaded_image = st.camera_input("Scan Product")

    if uploaded_image:
        image = Image.open(uploaded_image)
        # Perform prediction
        class_prediction = predict_image(model, image)
        matched_product = df.iloc[class_prediction]  # Assuming class indices match DataFrame rows
        
        st.write("**Visual Search Result:**")
        
        # Display image and product details in columns
        col1, col2 = st.columns([1,2])
        with col1:
            st.image(matched_product['image'], width=200)  # Adjust image size as needed
        with col2:
            st.write(f"**{matched_product['name']}**")
            st.write(f"Price: ${matched_product['price']}")
            if st.button("Add to Cart", key=f"add_{matched_product['id']}"):
                product_id = matched_product['id']
                if product_id in st.session_state.cart:
                    st.session_state.cart[product_id]['quantity'] += 1
                else:
                    st.session_state.cart[product_id] = {'product': matched_product, 'quantity': 1}
                st.success(f"{matched_product['name']} added to cart.")
                
            # If product is already in the cart, show "1 in cart - Remove" option
            product_id = matched_product['id']
            if product_id in st.session_state.cart:
                qty = st.session_state.cart[product_id]['quantity']
                st.write(f"{qty} in cart - ", end="")
                if st.button("Remove", key=f"remove_{product_id}"):
                    st.session_state.cart[product_id]['quantity'] -= 1
                    if st.session_state.cart[product_id]['quantity'] == 0:
                        del st.session_state.cart[product_id]
                    st.success(f"Removed one {matched_product['name']} from cart.")
                    
    if st.button("Back to Home"):
        st.session_state.visual_search = False

else:
    # Filter products based on search query
    if search_query:
        filtered_df = df[df['name'].str.contains(search_query, case=False, na=False)]
    else:
        filtered_df = df

    # Display filtered products with image on the left and details on the right
    if not filtered_df.empty:
        for index, product in filtered_df.iterrows():
            col1, col2 = st.columns([1,2])
            with col1:
                st.image(product['image'], width=200)  # Adjust image size as needed
            with col2:
                st.write(f"**{product['name']}**")
                st.write(f"Price: ${product['price']}")
                product_id = product['id']
                if st.button("Add to Cart", key=f"add_cart_{product_id}"):
                    if product_id in st.session_state.cart:
                        st.session_state.cart[product_id]['quantity'] += 1
                    else:
                        st.session_state.cart[product_id] = {'product': product, 'quantity': 1}
                    st.success(f"{product['name']} added to cart.")
                
                # Show "1 in cart - Remove" if the product is already in the cart
                if product_id in st.session_state.cart:
                    qty = st.session_state.cart[product_id]['quantity']
                    st.write(f"{qty} in cart - ", end="")
                    if st.button("Remove", key=f"remove_{product_id}"):
                        st.session_state.cart[product_id]['quantity'] -= 1
                        if st.session_state.cart[product_id]['quantity'] == 0:
                            del st.session_state.cart[product_id]
                        st.success(f"Removed one {product['name']} from cart.")
            
            st.markdown("---")  # Adds a horizontal line to separate products

    else:
        st.write("No products found matching your search.")

# Display the cart in the sidebar
st.sidebar.title("Cart")
if st.session_state.cart:
    for item in st.session_state.cart.values():
        st.sidebar.write(f"{item['product']['name']} - ${item['product']['price']} x {item['quantity']}")
    total = sum(item['product']['price'] * item['quantity'] for item in st.session_state.cart.values())
    st.sidebar.write(f"**Total: ${total:.2f}**")
else:
    st.sidebar.write("Your cart is empty.")

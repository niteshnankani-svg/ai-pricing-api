import streamlit as st
import requests
import plotly.graph_objects as go

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Pricing Prediction",
    page_icon="💰",
    layout="wide"
)

# ============================================
# HEADER
# ============================================
st.title("💰 AI Pricing Prediction Dashboard")
st.markdown("Predict optimal product prices using Machine Learning")
st.divider()

# ============================================
# INPUT FORM
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Details")
    product_name = st.text_input("Product Name", "Wireless Headphones")
    category = st.selectbox("Category", 
        ["Electronics", "Fitness", "Kitchen", "Fashion", "Books"])
    season = st.selectbox("Season", 
        ["regular", "holiday", "new_year"])

with col2:
    st.subheader("💲 Pricing Inputs")
    cost = st.number_input("Product Cost ($)", 
        min_value=1.0, max_value=1000.0, value=25.0, step=0.5)
    competitor_price = st.number_input("Competitor Price ($)", 
        min_value=1.0, max_value=2000.0, value=59.99, step=0.5)
    demand_score = st.slider("Demand Score", 
        min_value=1, max_value=10, value=8,
        help="1 = Very Low Demand, 10 = Very High Demand")

st.divider()

# ============================================
# PREDICT BUTTON
# ============================================
if st.button("🚀 Predict Price", type="primary", use_container_width=True):
    
    with st.spinner("Calling ML model..."):
        try:
            # Call your live API
            response = requests.post(
                "https://ai-pricing-api.onrender.com/predict",
                json={
                    "product_name": product_name,
                    "cost": cost,
                    "demand_score": demand_score,
                    "competitor_price": competitor_price,
                    "season": season
                },
                timeout=60
            )
            
            result = response.json()
            
            # ============================================
            # RESULTS
            # ============================================
            st.success("✅ Prediction Complete!")
            st.divider()
            
            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            
            m1.metric(
                "Predicted Price", 
                f"${result['predicted_price']}",
                delta=f"${round(result['predicted_price'] - cost, 2)} above cost"
            )
            m2.metric(
                "Monthly Revenue", 
                f"${result['predicted_revenue']:,}"
            )
            m3.metric(
                "Profit Margin", 
                f"{result['margin_percent']}%"
            )
            m4.metric(
                "Confidence", 
                result['confidence'].upper()
            )
            
            st.divider()
            
            # Price comparison chart
            st.subheader("📊 Price Analysis")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=["Your Cost", "Predicted Price", "Competitor Price"],
                    y=[cost, result['predicted_price'], competitor_price],
                    marker_color=["#ff4444", "#00cc44", "#4444ff"],
                    text=[f"${cost}", 
                          f"${result['predicted_price']}", 
                          f"${competitor_price}"],
                    textposition="outside"
                )
            ])
            
            fig.update_layout(
                title="Price Comparison",
                yaxis_title="Price ($)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Revenue breakdown
            st.subheader("📈 Revenue Breakdown")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.info(f"""
                **Product:** {product_name}
                
                **Estimated Units/Month:** {result['estimated_units']}
                
                **Monthly Revenue:** ${result['predicted_revenue']:,}
                
                **Annual Revenue:** ${result['predicted_revenue'] * 12:,.2f}
                """)
            
            with col_b:
                # Profit donut chart
                profit = result['predicted_price'] - cost
                fig2 = go.Figure(data=[go.Pie(
                    labels=["Profit", "Cost"],
                    values=[profit, cost],
                    hole=0.6,
                    marker_colors=["#00cc44", "#ff4444"]
                )])
                fig2.update_layout(
                    title="Cost vs Profit per Unit",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
            st.info("Note: First request may take 50 seconds as the API wakes up. Please try again!")

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("Built with Python · Flask · Scikit-learn · Streamlit · n8n")
```

---

**Step 3 — Create `requirements.txt`:**
```
streamlit
requests
plotly

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_and_train_models():
    #Load data
    df = pd.read_csv('digital_habits_vs_mental_health.csv')
    
    #Clean data
    df = df.dropna()
    valid = (
        (df['screen_time_hours'] >= 0) & (df['screen_time_hours'] <= 24) &
        (df['hours_on_TikTok'] >= 0) & (df['hours_on_TikTok'] <= df['screen_time_hours']) &
        (df['sleep_hours'] >= 0) & (df['sleep_hours'] <= 24) &
        (df['stress_level'].between(1, 10)) & (df['mood_score'].between(1, 10)) &
        (df['social_media_platforms_used'] >= 0)
    )
    df = df[valid].reset_index(drop=True)
    
    #Feature engineering
    df['stress_mood_index'] = (df['stress_level'] * 0.6) + ((11 - df['mood_score']) * 0.4)
    
    #Train models
    features = ['screen_time_hours', 'hours_on_TikTok', 'sleep_hours', 'stress_mood_index']
    
    #K-means
    X_clust = df[features]
    scaler_clust = StandardScaler()
    X_clust_scaled = scaler_clust.fit_transform(X_clust)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_clust_scaled)
    df['cluster'] = kmeans.labels_
    cluster_names = {0: 'Balanced Users', 1: 'At-Risk Digital Engagers', 2: 'Light & Healthy Users'}
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    #Random Forest
    y_clf = (df['stress_level'] > 7).astype(int)
    X_clf = df[features] 
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    scaler_clf = StandardScaler()
    X_train_scaled = scaler_clf.fit_transform(X_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
    
    cluster_summary = df.groupby('cluster')[features].mean()
    
    return df, scaler_clust, kmeans, scaler_clf, rf_model, features, cluster_summary, cluster_names

df, scaler_clust, kmeans, scaler_clf, rf_model, features, cluster_summary, cluster_names = load_and_train_models()

# Tips
def get_recommendation(cluster):
    recs = {
        0: "✅ **Balanced Users**: Keep up great habits! Maintain moderate usage and good sleep.",
        1: "⚠️ **At-Risk Digital Engagers**: Reduce Screen Time (TikTok) esp. before bed and aim for 6.5+ hrs sleep.",
        2: "🌟 **Light & Healthy Users**: Excellent! Share your routine as a wellness role model."
    }
    return recs.get(cluster, "No recommendation available.")

st.set_page_config(page_title="Digital Habits & Mental Health", layout="wide")
st.title("📱 Digital Habits & Mental Health Dashboard")
st.markdown("""
Enter your habits below to get personalized insights based on data from **98,135 working adults**.
""")

#Sidebar
with st.sidebar:
    st.header("📊 Population Insights")
    st.metric("Avg Screen Time", f"{df['screen_time_hours'].mean():.1f} hrs")
    st.metric("Avg Sleep", f"{df['sleep_hours'].mean():.1f} hrs")
    st.metric("High-Stress Users", f"{(df['stress_level'] > 7).mean()*100:.1f}%")
    
    st.subheader("User Segments")
    st.bar_chart(df['cluster_name'].value_counts())

#USER INPUT
st.header("📝 Your Digital Habits")
col1, col2 = st.columns(2)

with col1:
    screen_time = st.number_input(
        'Daily Screen Time (hours)', 
        min_value=0.0, max_value=24.0, value=5.0, step=0.1
    )
    
with col2:
    # Enforce TikTok ≤ Screen Time
    max_tiktok = screen_time
    tiktok_time = st.number_input(
        'Daily Hours on TikTok', 
        min_value=0.0, max_value=max_tiktok, value=min(1.0, max_tiktok), step=0.1,
        help="TikTok time cannot exceed total screen time."
    )

col3, col4 = st.columns(2)
with col3:
    sleep_time = st.number_input('Daily Sleep Hours', min_value=0.0, max_value=24.0, value=7.0, step=0.1)
with col4:
    stress_level = st.slider('Stress Level (1-10)', min_value=1, max_value=10, value=5)
    mood_score = st.slider('Mood Score (1-10)', min_value=1, max_value=10, value=7)

if st.button('🔍 Analyze My Habits', type="primary"):
    # Validate TikTok ≤ Screen Time
    if tiktok_time > screen_time:
        st.error("❌ TikTok hours cannot exceed total screen time. Please adjust.")
        st.stop()
    
    stress_mood_index = (stress_level * 0.6) + ((11 - mood_score) * 0.4)
    
    user_input_clust = np.array([[screen_time, tiktok_time, sleep_time, stress_mood_index]])
    user_input_clust_scaled = scaler_clust.transform(user_input_clust)
    user_cluster = kmeans.predict(user_input_clust_scaled)[0]
    user_cluster_name = cluster_names[user_cluster]
    
    user_input_clf_scaled = scaler_clf.transform(user_input_clust)
    high_stress_prob = rf_model.predict_proba(user_input_clf_scaled)[0, 1]
    
    #RESULTS
    st.header("🎯 Your Personalized Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Stress-Mood Index", f"{stress_mood_index:.2f}")
    res_col2.metric("User Type", user_cluster_name)
    res_col3.metric("High-Stress Risk", f"{high_stress_prob:.0%}")
    
    st.markdown(get_recommendation(user_cluster))
    
    #VISUALIZATIONS
    st.header("📈 Visual Insights")
    
    #Random Forest
    st.subheader("What Drives High-Stress Risk?")
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    fig_imp, ax_imp = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax_imp, palette='Blues_d')
    ax_imp.set_title('Feature Importance for High-Stress Prediction')
    st.pyplot(fig_imp)
    
    #You vs. Your Cluster
    st.subheader("You vs. Your Cluster Average")
    cluster_vals = cluster_summary.loc[user_cluster]
    user_vals = [screen_time, tiktok_time, sleep_time, stress_mood_index]
    
    # Normalize for radar
    min_vals = df[features].min()
    max_vals = df[features].max()
    norm_user = (np.array(user_vals) - min_vals) / (max_vals - min_vals)
    norm_cluster = (cluster_vals - min_vals) / (max_vals - min_vals)
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    norm_user = np.concatenate((norm_user, [norm_user[0]]))
    norm_cluster = np.concatenate((norm_cluster, [norm_cluster[0]]))
    angles += angles[:1]
    
    fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, norm_user, 'o-', linewidth=2, label='You')
    ax_radar.fill(angles, norm_user, alpha=0.25)
    ax_radar.plot(angles, norm_cluster, 'o-', linewidth=2, label=f'Your Cluster Avg ({user_cluster_name})')
    ax_radar.fill(angles, norm_cluster, alpha=0.25)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), features)
    ax_radar.set_ylim(0, 1)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig_radar)
    
    #SCATTER PLOT
    st.subheader("Your Position in the Population")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
    
    # Plot all users
    scatter = ax_scatter.scatter(
        df['hours_on_TikTok'], df['stress_mood_index'],
        c=df['cluster'], cmap='viridis', alpha=0.5, s=10
    )
    # Highlight user
    ax_scatter.scatter(tiktok_time, stress_mood_index, color='red', s=200, label='You', edgecolor='black')
    ax_scatter.set_xlabel('Hours on TikTok')
    ax_scatter.set_ylabel('Stress-Mood Index')
    ax_scatter.set_title('TikTok Usage vs. Mental Health Burden')
    ax_scatter.legend()
    st.pyplot(fig_scatter)
    
    #EXAMPLES FROM SAME CLUSTER
    st.header("👥 Real Examples from Your Cluster")
    st.write(f"You’re in the **{user_cluster_name}** group. Here are 10 real users like you:")
    
    same_cluster_examples = df[df['cluster'] == user_cluster].sample(
        n=min(10, len(df[df['cluster'] == user_cluster])), 
        random_state=42
    )
    
    example_df = same_cluster_examples[['screen_time_hours', 'hours_on_TikTok', 'sleep_hours', 'stress_level', 'mood_score']].copy()
    example_df.columns = ['Screen Time (hrs)', 'TikTok (hrs)', 'Sleep (hrs)', 'Stress', 'Mood']
    example_df.index = [f"User {i+1}" for i in range(len(example_df))]
    
    st.dataframe(example_df.style.format("{:.1f}"))
    st.caption("💡 These are real responses from others in your segment. Yours may differ!")

st.markdown("---")
st.caption("💡 **Try this**: Reduce TikTok usage and increase sleep, see how your metrics improve!")
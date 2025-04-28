# Load sample data for visualization
df = load_sample_data()
# Add tab-based navigation for different visualizations
viz_tabs = st.tabs(["Interactive Explorer", "Key Factors", "Demographic Analysis", "Regional Trends", "Correlation Matrix"])

with viz_tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Interactive Insurance Cost Explorer")
    
    # Create flexible visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("X-Axis", options=["age", "bmi", "children", "charges"], index=0)
        color_by = st.selectbox("Color By", options=["smoker", "sex", "region", "None"], index=0)
    
    with col2:
        y_axis = st.selectbox("Y-Axis", options=["charges", "age", "bmi", "children"], index=0)
        size_by = st.selectbox("Size By", options=["None", "age", "bmi", "children"], index=0)
        
        # Add advanced visualization option
        chart_type = st.selectbox("Chart Type", options=[
            "Scatter Plot", 
            "Bubble Chart",
            "3D Scatter",
            "Violin Plot",
            "Box Plot",
            "Density Contour",
            "Heat Map",  # New chart type
            "Line Plot",  # New chart type
            "Area Plot"   # New chart type
        ])
    
    # Handle "None" options
    color_col = None if color_by == "None" else color_by
    size_col = None if size_by == "None" else size_by
    
    # Create conditional layout based on chart type
    if chart_type in ["Scatter Plot", "Bubble Chart", "Heat Map", "Line Plot", "Area Plot"]:
        # Advanced configuration for 2D plots
        advanced_options = st.expander("Advanced Options")
        with advanced_options:
            trend_line = st.checkbox("Show Trend Line", value=True)
            log_scale = st.checkbox("Logarithmic Scale", value=False)
            facet_by = st.selectbox("Facet By", options=["None", "smoker", "sex", "region"], index=0)
            opacity = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
            
            # Animation option (new feature)
            if chart_type in ["Scatter Plot", "Bubble Chart"]:
                enable_animation = st.checkbox("Enable Animation", value=False)
                if enable_animation:
                    animation_col = st.selectbox("Animate By", options=["age", "bmi", "children"], index=0)
            
            # Color scheme selection
            color_scheme = st.selectbox("Color Scheme", options=[
                "Blues", "Reds", "Greens", "Viridis", "Plasma", "Turbo", 
                "Spectral", "RdYlBu", "YlOrRd"  # New color schemes
            ])
    
    # Export options (new feature)
    export_options = st.expander("Export Options")
    with export_options:
        export_format = st.selectbox("Export Format", ["PNG", "JPEG", "SVG", "PDF", "HTML"], index=0)
        export_width = st.number_input("Width (px)", min_value=800, max_value=3000, value=1200)
        export_height = st.number_input("Height (px)", min_value=600, max_value=2500, value=800)
    
    # Render different chart types
    if chart_type == "Scatter Plot":
        fig = px.scatter(
            df, x=x_axis, y=y_axis, 
            color=color_col, 
            hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
            title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
            height=600,
            opacity=opacity,
            log_y=log_scale if y_axis == "charges" else False,
            log_x=log_scale if x_axis == "charges" else False,
            trendline="ols" if trend_line else None,
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
            color_continuous_scale=color_scheme
        )
        
        # Apply faceting if selected
        if facet_by != "None":
            fig = px.scatter(
                df, x=x_axis, y=y_axis, 
                color=color_col,
                facet_col=facet_by,
                hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()} by {facet_by.capitalize()}",
                height=600,
                opacity=opacity,
                log_y=log_scale if y_axis == "charges" else False,
                log_x=log_scale if x_axis == "charges" else False,
                trendline="ols" if trend_line else None,
                color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                color_continuous_scale=color_scheme
            )
    
    elif chart_type == "Bubble Chart":
        # For bubble chart, we need a size parameter
        size_value = size_col if size_col else "bmi"
        
        fig = px.scatter(
            df, x=x_axis, y=y_axis, 
            color=color_col,
            size=size_value,
            hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
            title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}, Size by {size_value.capitalize()}",
            height=600,
            opacity=opacity,
            log_y=log_scale if y_axis == "charges" else False,
            log_x=log_scale if x_axis == "charges" else False,
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
            color_continuous_scale=color_scheme
        )
    
    elif chart_type == "3D Scatter":
        # For 3D scatter, we need a z-axis
        z_axis = st.selectbox("Z-Axis", options=["charges", "age", "bmi", "children"], index=0)
        
        fig = px.scatter_3d(
            df, x=x_axis, y=y_axis, z=z_axis,
            color=color_col,
            size=size_col if size_col else None,
            hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
            title=f"3D View: {x_axis.capitalize()} vs {y_axis.capitalize()} vs {z_axis.capitalize()}",
            opacity=opacity,
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
            color_continuous_scale=color_scheme
        )
        
        fig.update_layout(height=700)
    
    elif chart_type == "Violin Plot":
        # For violin plot, x should be categorical and y numeric
        if x_axis in ["sex", "smoker", "region"]:
            fig = px.violin(
                df, x=x_axis, y=y_axis,
                color=color_col,
                box=True,
                points="all",
                hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                title=f"Distribution of {y_axis.capitalize()} by {x_axis.capitalize()}",
                color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                color_continuous_scale=color_scheme
            )
        else:
            # If x is numeric, we need to bin it
            if x_axis == "age":
                # Create temporary age groups
                temp_df = df.copy()
                temp_df['age_group'] = pd.cut(
                    temp_df['age'], 
                    bins=[0, 20, 30, 40, 50, 60, 100],
                    labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
                )
                
                fig = px.violin(
                    temp_df, x='age_group', y=y_axis,
                    color=color_col,
                    box=True,
                    points="all",
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by Age Group",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
            elif x_axis == "bmi":
                # Create temporary BMI categories
                temp_df = df.copy()
                temp_df['bmi_category'] = pd.cut(
                    temp_df['bmi'],
                    bins=[0, 18.5, 25, 30, 35, 40, 100],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
                )
                
                fig = px.violin(
                    temp_df, x='bmi_category', y=y_axis,
                    color=color_col,
                    box=True,
                    points="all",
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by BMI Category",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
            else:
                # Use numeric x-axis with fewer points (like children)
                fig = px.violin(
                    df, x=x_axis, y=y_axis,
                    color=color_col,
                    box=True,
                    points="all",
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by {x_axis.capitalize()}",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
    
    elif chart_type == "Box Plot":
        # Similar approach as violin plot for categorical x-axis
        if x_axis in ["sex", "smoker", "region"]:
            fig = px.box(
                df, x=x_axis, y=y_axis,
                color=color_col,
                points="all",
                notched=True,  # Add notches to box plot
                hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                title=f"Distribution of {y_axis.capitalize()} by {x_axis.capitalize()}",
                color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                color_continuous_scale=color_scheme
            )
        else:
            # If x is numeric, we need to bin it
            if x_axis == "age":
                # Create temporary age groups
                temp_df = df.copy()
                temp_df['age_group'] = pd.cut(
                    temp_df['age'], 
                    bins=[0, 20, 30, 40, 50, 60, 100],
                    labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
                )
                
                fig = px.box(
                    temp_df, x='age_group', y=y_axis,
                    color=color_col,
                    points="all",
                    notched=True,
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by Age Group",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
            elif x_axis == "bmi":
                # Create temporary BMI categories
                temp_df = df.copy()
                temp_df['bmi_category'] = pd.cut(
                    temp_df['bmi'],
                    bins=[0, 18.5, 25, 30, 35, 40, 100],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
                )
                
                fig = px.box(
                    temp_df, x='bmi_category', y=y_axis,
                    color=color_col,
                    points="all",
                    notched=True,
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by BMI Category",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
            else:
                # Use numeric x-axis with fewer points (like children)
                fig = px.box(
                    df, x=x_axis, y=y_axis,
                    color=color_col,
                    points="all",
                    notched=True,
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by {x_axis.capitalize()}",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
    
    elif chart_type == "Density Contour":
        fig = px.density_contour(
            df, x=x_axis, y=y_axis,
            color=color_col,
            marginal_x="histogram",
            marginal_y="histogram",
            title=f"Density Contour of {y_axis.capitalize()} vs {x_axis.capitalize()}",
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
            color_continuous_scale=color_scheme
        )
    
    # New chart type: Heat Map
    elif chart_type == "Heat Map":
        # For heatmap, we need to aggregate the data
        if x_axis in ["age", "bmi"]:
            # Bin continuous variables
            if x_axis == "age":
                temp_df = df.copy()
                temp_df['age_group'] = pd.cut(
                    temp_df['age'], 
                    bins=[18, 25, 35, 45, 55, 65, 100],
                    labels=['18-25', '25-35', '35-45', '45-55', '55-65', '65+']
                )
                heatmap_df = temp_df.groupby(['age_group', y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'bmi_category']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                # Create a pivot table
                pivot_table = heatmap_df.pivot_table(
                    values='charges', 
                    index='age_group', 
                    columns=y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'bmi_category'
                )
                
                fig = px.imshow(
                    pivot_table,
                    labels=dict(x=y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'BMI Category', 
                                y='Age Group', 
                                color='Average Cost ($)'),
                    title=f"Average Insurance Cost by Age Group and {y_axis.capitalize() if y_axis in ['children', 'sex', 'smoker', 'region'] else 'BMI Category'}",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=color_scheme
                )
                
                # Format text as currency
                fig.update_traces(texttemplate="$%{z:.0f}")
                
            elif x_axis == "bmi":
                temp_df = df.copy()
                temp_df['bmi_category'] = pd.cut(
                    temp_df['bmi'],
                    bins=[0, 18.5, 25, 30, 35, 40, 100],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
                )
                
                heatmap_df = temp_df.groupby(['bmi_category', y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'age_group']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                # Create a pivot table
                pivot_table = heatmap_df.pivot_table(
                    values='charges', 
                    index='bmi_category', 
                    columns=y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'age_group'
                )
                
                fig = px.imshow(
                    pivot_table,
                    labels=dict(x=y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'Age Group', 
                                y='BMI Category', 
                                color='Average Cost ($)'),
                    title=f"Average Insurance Cost by BMI Category and {y_axis.capitalize() if y_axis in ['children', 'sex', 'smoker', 'region'] else 'Age Group'}",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=color_scheme
                )
                
                # Format text as currency
                fig.update_traces(texttemplate="$%{z:.0f}")
                
        else:
            # Use categorical variables directly
            if x_axis in ["sex", "smoker", "region"] and y_axis in ["sex", "smoker", "region", "children"]:
                heatmap_df = df.groupby([x_axis, y_axis]).agg({
                    'charges': 'mean'
                }).reset_index()
                
                # Create a pivot table
                pivot_table = heatmap_df.pivot_table(values='charges', index=x_axis, columns=y_axis)
                
                fig = px.imshow(
                    pivot_table,
                    labels=dict(x=y_axis.capitalize(), y=x_axis.capitalize(), color='Average Cost ($)'),
                    title=f"Average Insurance Cost by {x_axis.capitalize()} and {y_axis.capitalize()}",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=color_scheme
                )
                
                # Format text as currency
                fig.update_traces(texttemplate="$%{z:.0f}")
            else:
                # For other combinations, just show a message
                st.warning("Heat maps work best with categorical variables or binned numeric variables")
                # Show a simple scatter plot instead
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, 
                    color=color_col,
                    title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                    height=600,
                    opacity=opacity,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
    
    # New chart type: Line Plot
    elif chart_type == "Line Plot":
        # For line plots, we need to aggregate data
        if x_axis == "age":
            # Group by age and calculate mean
            line_df = df.groupby('age').agg({
                'charges': 'mean'
            }).reset_index()
            
            fig = px.line(
                line_df, x='age', y='charges',
                title=f"Average Insurance Cost by Age",
                labels={'charges': 'Average Cost ($)', 'age': 'Age'},
                markers=True,
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            # Add smoker vs non-smoker line if color is set to smoker
            if color_col == "smoker":
                smoker_line_df = df.groupby(['age', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    smoker_line_df, x='age', y='charges', color='smoker',
                    title=f"Average Insurance Cost by Age and Smoking Status",
                    labels={'charges': 'Average Cost ($)', 'age': 'Age', 'smoker': 'Smoker'},
                    markers=True,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
            # Add gender lines if color is set to sex
            elif color_col == "sex":
                gender_line_df = df.groupby(['age', 'sex']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    gender_line_df, x='age', y='charges', color='sex',
                    title=f"Average Insurance Cost by Age and Gender",
                    labels={'charges': 'Average Cost ($)', 'age': 'Age', 'sex': 'Gender'},
                    markers=True,
                    color_discrete_map={"male": "#3498db", "female": "#9b59b6"}
                )
            # Add region lines if color is set to region
            elif color_col == "region":
                region_line_df = df.groupby(['age', 'region']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    region_line_df, x='age', y='charges', color='region',
                    title=f"Average Insurance Cost by Age and Region",
                    labels={'charges': 'Average Cost ($)', 'age': 'Age', 'region': 'Region'},
                    markers=True
                )
        
        elif x_axis == "bmi":
            # Group by BMI (rounded) and calculate mean
            temp_df = df.copy()
            temp_df['bmi_rounded'] = round(temp_df['bmi'])
            
            line_df = temp_df.groupby('bmi_rounded').agg({
                'charges': 'mean'
            }).reset_index()
            
            fig = px.line(
                line_df, x='bmi_rounded', y='charges',
                title=f"Average Insurance Cost by BMI",
                labels={'charges': 'Average Cost ($)', 'bmi_rounded': 'BMI'},
                markers=True,
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            # Add smoker vs non-smoker line if color is set to smoker
            if color_col == "smoker":
                smoker_line_df = temp_df.groupby(['bmi_rounded', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    smoker_line_df, x='bmi_rounded', y='charges', color='smoker',
                    title=f"Average Insurance Cost by BMI and Smoking Status",
                    labels={'charges': 'Average Cost ($)', 'bmi_rounded': 'BMI', 'smoker': 'Smoker'},
                    markers=True,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
        
        else:
            # For other x-axes, show a simple line chart
            if x_axis in ["children", "sex", "smoker", "region"]:
                line_df = df.groupby(x_axis).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    line_df, x=x_axis, y='charges',
                    title=f"Average Insurance Cost by {x_axis.capitalize()}",
                    labels={'charges': 'Average Cost ($)', x_axis: x_axis.capitalize()},
                    markers=True,
                    color_discrete_sequence=[px.colors.sequential.Plasma[4]]
                )
            else:
                # Default to scatter plot if x-axis doesn't make sense for a line chart
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, 
                    color=color_col,
                    title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                    height=600,
                    opacity=opacity,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
    
    # New chart type: Area Plot
    elif chart_type == "Area Plot":
        if x_axis == "age":
            # Group by age and calculate mean
            area_df = df.groupby('age').agg({
                'charges': 'mean'
            }).reset_index()
            
            fig = px.area(
                area_df, x='age', y='charges',
                title=f"Average Insurance Cost by Age",
                labels={'charges': 'Average Cost ($)', 'age': 'Age'},
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            # Add smoker vs non-smoker areas if color is set to smoker
            if color_col == "smoker":
                smoker_area_df = df.groupby(['age', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.area(
                    smoker_area_df, x='age', y='charges', color='smoker',
                    title=f"Average Insurance Cost by Age and Smoking Status",
                    labels={'charges': 'Average Cost ($)', 'age': 'Age', 'smoker': 'Smoker'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
        elif x_axis == "bmi":
            # Group by BMI (rounded) and calculate mean
            temp_df = df.copy()
            temp_df['bmi_rounded'] = round(temp_df['bmi'])
            
            area_df = temp_df.groupby('bmi_rounded').agg({
                'charges': 'mean'
            }).reset_index()
            
            fig = px.area(
                area_df, x='bmi_rounded', y='charges',
                title=f"Average Insurance Cost by BMI",
                labels={'charges': 'Average Cost ($)', 'bmi_rounded': 'BMI'},
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            # Add smoker vs non-smoker areas if color is set to smoker
            if color_col == "smoker":
                smoker_area_df = temp_df.groupby(['bmi_rounded', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.area(
                    smoker_area_df, x='bmi_rounded', y='charges', color='smoker',
                    title=f"Average Insurance Cost by BMI and Smoking Status",
                    labels={'charges': 'Average Cost ($)', 'bmi_rounded': 'BMI', 'smoker': 'Smoker'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
        else:
            # Default to scatter plot for other x-axes
            fig = px.scatter(
                df, x=x_axis, y=y_axis, 
                color=color_col,
                title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                height=600,
                opacity=opacity,
                color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                color_continuous_scale=color_scheme
            )
    
    # Apply logarithmic scales if requested
    if chart_type not in ["Heat Map", "Violin Plot", "Box Plot"]:
        if log_scale and y_axis == "charges":
            fig.update_yaxes(type="log")
        if log_scale and x_axis == "charges":
            fig.update_xaxes(type="log")
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights based on chart
    insight_expander = st.expander("Chart Insights")
    with insight_expander:
        if x_axis == "age" and y_axis == "charges":
            st.info("🔍 **Insight**: There's a positive correlation between age and insurance costs. Premiums tend to increase with age as health risks rise.")
            
            # Calculate correlation
            correlation = df[['age', 'charges']].corr().iloc[0, 1]
            st.metric("Age-Cost Correlation", f"{correlation:.2f}")
            
            # Show age group averages
            age_group_avgs = df.groupby('age_group')['charges'].mean().reset_index()
            age_group_avgs['charges'] = age_group_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by age group:")
            st.dataframe(age_group_avgs)
            
        elif x_axis == "bmi" and y_axis == "charges":
            st.info("🔍 **Insight**: BMI shows a moderate positive correlation with insurance costs. Higher BMI values are associated with greater health risks and higher premiums.")
            
            # Calculate correlation
            correlation = df[['bmi', 'charges']].corr().iloc[0, 1]
            st.metric("BMI-Cost Correlation", f"{correlation:.2f}")
            
            # Show BMI category averages
            bmi_cat_avgs = df.groupby('bmi_category')['charges'].mean().reset_index()
            bmi_cat_avgs['charges'] = bmi_cat_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by BMI category:")
            st.dataframe(bmi_cat_avgs)
            
        elif color_col == "smoker" and y_axis == "charges":
            st.warning("🔥 **Key Finding**: Smoking has the strongest impact on insurance costs among all factors. The vertical separation between smokers and non-smokers is substantial at all age and BMI levels.")
            
            # Show smoker vs non-smoker average
            smoker_avgs = df.groupby('smoker')['charges'].mean().reset_index()
            smoker_avgs['charges'] = smoker_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by smoking status:")
            st.dataframe(smoker_avgs)
            
        elif x_axis == "children" and y_axis == "charges":
            st.info("🔍 **Insight**: The number of dependents covered has a modest effect on insurance premiums. There's a slight positive correlation between family size and costs.")
            
            # Calculate correlation
            correlation = df[['children', 'charges']].corr().iloc[0, 1]
            st.metric("Dependents-Cost Correlation", f"{correlation:.2f}")
            
        elif (x_axis == "region" or color_col == "region") and y_axis == "charges":
            st.info("🔍 **Insight**: Geographic region can affect insurance costs due to local healthcare prices, regulations, and population health factors.")
            
            # Show region averages
            region_avgs = df.groupby('region')['charges'].mean().reset_index()
            region_avgs['charges'] = region_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by region:")
            st.dataframe(region_avgs)
            
        elif (x_axis == "sex" or color_col == "sex") and y_axis == "charges":
            st.info("🔍 **Insight**: Gender differences in insurance costs are relatively minor compared to factors like age and smoking status.")
            
            # Show gender averages
            gender_avgs = df.groupby('sex')['charges'].mean().reset_index()
            gender_avgs['charges'] = gender_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by gender:")
            st.dataframe(gender_avgs)
        
        # Add data summary statistics section
        st.subheader("Summary Statistics")
        if y_axis in ["charges", "age", "bmi", "children"]:
            stats_df = df[y_axis].describe().reset_index()
            stats_df.columns = ["Statistic", "Value"]
            
            # Format values for currency if charges
            if y_axis == "charges":
                stats_df["Value"] = stats_df["Value"].map('${:,.2f}'.format)
            else:
                stats_df["Value"] = stats_df["Value"].round(2)
                
            st.dataframe(stats_df)
    
    # Add download options
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Download the visualization or data for your reports")
    
    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            label="📊 Download Chart",
            data="This feature is not available in the example code.",
            file_name=f"insurance_chart_{chart_type}.png",
            mime="image/png",
            disabled=True
        )
    
    with download_cols[1]:
        st.download_button(
            label="📄 Download Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="insurance_data.csv",
            mime="text/csv"
        )
        
    st.markdown("</div>", unsafe_allow_html=True)

# New implementation for Demographic Analysis tab
with viz_tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Demographic Insurance Cost Analysis")
    
    # Create a dashboard of demographic insights
    demo_tabs = st.tabs(["Age Analysis", "Gender Analysis", "BMI Impact", "Family Size", "Combined Factors"])
    
    with demo_tabs[0]:
        st.markdown("### Age and Insurance Costs")
        
        # Create age groups for better analysis - use numeric values for calculations
        df['age_numeric'] = df['age']  # Keep numeric version for calculations
        
        # Create string age groups for visualization
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 20, 30, 40, 50, 60, 100],
            labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
        )
        
        # Age group statistics - use numeric age for calculations
        age_stats = df.groupby('age_group').agg({
            'charges': ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        age_stats.columns = ['Age Group', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
        
        # Format monetary columns
        age_stats['Mean Cost'] = age_stats['Mean Cost'].map('${:,.2f}'.format)
        age_stats['Median Cost'] = age_stats['Median Cost'].map('${:,.2f}'.format)
        age_stats['Std Dev'] = age_stats['Std Dev'].map('${:,.2f}'.format)
        
        # Display age statistics
        st.dataframe(age_stats, use_container_width=True)
        
        # Create age trend visualization
        fig = px.line(
            df.groupby('age_numeric').agg({'charges': 'mean'}).reset_index(),
            x='age_numeric',
            y='charges',
            title='Average Insurance Cost by Age',
            labels={'charges': 'Average Cost ($)', 'age_numeric': 'Age'},
            markers=True
        )
        
        # Add smoker vs non-smoker trend lines
        smoker_age_data = df.groupby(['age_numeric', 'smoker']).agg({'charges': 'mean'}).reset_index()
        
        fig2 = px.line(
            smoker_age_data,
            x='age_numeric',
            y='charges',
            color='smoker',
            title='Average Insurance Cost by Age and Smoking Status',
            labels={'charges': 'Average Cost ($)', 'age_numeric': 'Age'},
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"},
            markers=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Add average cost increase per year
        age_increase = st.expander("Age Cost Analysis")
        with age_increase:
            # Calculate avg cost increase per year of age
            age_costs = df.groupby('age_numeric')['charges'].mean().reset_index()
            
            # Calculate the average increase per year
            age_costs['next_age_cost'] = age_costs['charges'].shift(-1)
            age_costs['cost_increase'] = age_costs['next_age_cost'] - age_costs['charges']
            
            avg_increase_per_year = age_costs['cost_increase'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Cost Increase Per Year of Age", f"${avg_increase_per_year:.2f}")
            
            with col2:
                # Calculate the percentage increase from youngest to oldest
                youngest_cost = age_costs.iloc[0]['charges']
                oldest_cost = age_costs.iloc[-2]['charges']  # -2 to avoid NaN in the last row
                pct_increase = ((oldest_cost - youngest_cost) / youngest_cost) * 100
                
                st.metric("Percentage Increase from Youngest to Oldest", f"{pct_increase:.1f}%")
            
            # Show a bar chart of cost increase by decade
            decade_bins = [18, 30, 40, 50, 60, 70, 100]
            decade_labels = ['18-30', '30-40', '40-50', '50-60', '60-70', '70+']
            
            df['decade'] = pd.cut(df['age'], bins=decade_bins, labels=decade_labels)
            decade_costs = df.groupby('decade')['charges'].mean().reset_index()
            
            fig = px.bar(
                decade_costs,
                x='decade',
                y='charges',
                title='Average Insurance Cost by Age Decade',
                labels={'charges': 'Average Cost ($)', 'decade': 'Age Group'},
                text_auto='.0f',
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            fig.update_traces(texttemplate='$%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Age-related insights
        st.markdown("""
        <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
            <h4 style="margin-top: 0; color: #3498db;">Age Insights</h4>
            <ul>
                <li><strong>Linear Growth:</strong> Insurance costs generally increase linearly with age</li>
                <li><strong>Higher Variance:</strong> Older age groups show greater variability in costs</li>
                <li><strong>Accelerated Increase:</strong> Costs rise more steeply after age 50</li>
                <li><strong>Smoking Amplifier:</strong> The cost increase with age is much steeper for smokers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # The rest of your Demographic Analysis tab code...
    with demo_tabs[1]:
        st.markdown("### Gender Analysis")
        
        # Gender statistics
        gender_stats = df.groupby('sex').agg({
            'charges': ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        gender_stats.columns = ['Gender', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
        
        # Format monetary columns
        gender_stats['Mean Cost'] = gender_stats['Mean Cost'].map('${:,.2f}'.format)
        gender_stats['Median Cost'] = gender_stats['Median Cost'].map('${:,.2f}'.format)
        gender_stats['Std Dev'] = gender_stats['Std Dev'].map('${:,.2f}'.format)
        
        # Gender and age analysis - use age_group string column for display
        gender_age = df.groupby(['sex', 'age_group']).agg({
            'charges': 'mean'
        }).reset_index()
        
        # Create visualizations
        fig1 = px.bar(
            df.groupby('sex').agg({'charges': 'mean'}).reset_index(),
            x='sex',
            y='charges',
            color='sex',
            title='Average Insurance Cost by Gender',
            labels={'charges': 'Average Cost ($)', 'sex': 'Gender'},
            color_discrete_map={"male": "#3498db", "female": "#9b59b6"},
            text_auto='.2f'
        )
        
        fig1.update_traces(texttemplate='$%{text}', textposition='outside')
        
        fig2 = px.bar(
            gender_age,
            x='age_group',
            y='charges',
            color='sex',
            barmode='group',
            title='Average Insurance Cost by Gender and Age Group',
            labels={'charges': 'Average Cost ($)', 'age_group': 'Age Group', 'sex': 'Gender'},
            color_discrete_map={"male": "#3498db", "female": "#9b59b6"}
        )
        
        # Display gender statistics and charts
        st.dataframe(gender_stats, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Enhanced gender analysis
        gender_analysis = st.expander("Detailed Gender Analysis")
        with gender_analysis:
            # Calculate gender disparity by different factors
            st.subheader("Gender Cost Differences by Factor")
            
            # By smoking status
            gender_smoking_df = df.groupby(['sex', 'smoker']).agg({
                'charges': 'mean'
            }).reset_index()
            
            gender_smoking_pivot = gender_smoking_df.pivot(index='smoker', columns='sex', values='charges')
            gender_smoking_pivot['difference'] = gender_smoking_pivot['male'] - gender_smoking_pivot['female']
            gender_smoking_pivot['pct_difference'] = (gender_smoking_pivot['difference'] / gender_smoking_pivot['female']) * 100
            
            gender_smoking_pivot['male'] = gender_smoking_pivot['male'].map('${:,.2f}'.format)
            gender_smoking_pivot['female'] = gender_smoking_pivot['female'].map('${:,.2f}'.format)
            gender_smoking_pivot['difference'] = gender_smoking_pivot['difference'].map('${:,.2f}'.format)
            gender_smoking_pivot['pct_difference'] = gender_smoking_pivot['pct_difference'].round(1).astype(str) + '%'
            
            gender_smoking_pivot = gender_smoking_pivot.reset_index()
            gender_smoking_pivot.columns = ['Smoking Status', 'Male Cost', 'Female Cost', 'Cost Difference', 'Percentage Difference']
            
            st.write("Gender difference by smoking status:")
            st.dataframe(gender_smoking_pivot, use_container_width=True)
            
            # By BMI category
            gender_bmi_df = df.groupby(['sex', 'bmi_category']).agg({
                'charges': 'mean'
            }).reset_index()
            
            fig3 = px.bar(
                gender_bmi_df,
                x='bmi_category',
                y='charges',
                color='sex',
                barmode='group',
                title='Average Insurance Cost by Gender and BMI Category',
                labels={'charges': 'Average Cost ($)', 'bmi_category': 'BMI Category', 'sex': 'Gender'},
                color_discrete_map={"male": "#3498db", "female": "#9b59b6"}
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Gender and smoking status
        gender_smoking = df.groupby(['sex', 'smoker']).agg({
            'charges': ['mean', 'count']
        }).reset_index()
        
        gender_smoking.columns = ['Gender', 'Smoker', 'Average Cost', 'Count']
        gender_smoking['Average Cost'] = gender_smoking['Average Cost'].map('${:,.2f}'.format)
        
        st.subheader("Gender and Smoking Status")
        st.dataframe(gender_smoking, use_container_width=True)
        
        # Gender insights
        st.markdown("""
        <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
            <h4 style="margin-top: 0; color: #3498db;">Gender Insights</h4>
            <ul>
                <li><strong>Small Difference:</strong> Gender alone is a relatively minor factor in insurance costs</li>
                <li><strong>Age Interaction:</strong> Gender differences vary by age group</li>
                <li><strong>Smoking Impact:</strong> Smoking status has a much larger effect than gender</li>
                <li><strong>Combined Factors:</strong> When combined with other risk factors, gender may have more significant effects</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with demo_tabs[2]:
        st.markdown("### BMI Impact Analysis")
        
        # Create BMI categories
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
        )
        
        # BMI statistics
        bmi_stats = df.groupby('bmi_category').agg({
            'charges# Load sample data for visualization
df = load_sample_data()
# Add tab-based navigation for different visualizations
viz_tabs = st.tabs(["Interactive Explorer", "Key Factors", "Demographic Analysis", "Regional Trends", "Correlation Matrix"])

with viz_tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Interactive Insurance Cost Explorer")
    
    # Create flexible visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("X-Axis", options=["age", "bmi", "children", "charges"], index=0)
        color_by = st.selectbox("Color By", options=["smoker", "sex", "region", "None"], index=0)
    
    with col2:
        y_axis = st.selectbox("Y-Axis", options=["charges", "age", "bmi", "children"], index=0)
        size_by = st.selectbox("Size By", options=["None", "age", "bmi", "children"], index=0)
        
        # Add advanced visualization option
        chart_type = st.selectbox("Chart Type", options=[
            "Scatter Plot", 
            "Bubble Chart",
            "3D Scatter",
            "Violin Plot",
            "Box Plot",
            "Density Contour",
            "Heat Map",  # New chart type
            "Line Plot",  # New chart type
            "Area Plot"   # New chart type
        ])
    
    # Handle "None" options
    color_col = None if color_by == "None" else color_by
    size_col = None if size_by == "None" else size_by
    
    # Create conditional layout based on chart type
    if chart_type in ["Scatter Plot", "Bubble Chart", "Heat Map", "Line Plot", "Area Plot"]:
        # Advanced configuration for 2D plots
        advanced_options = st.expander("Advanced Options")
        with advanced_options:
            trend_line = st.checkbox("Show Trend Line", value=True)
            log_scale = st.checkbox("Logarithmic Scale", value=False)
            facet_by = st.selectbox("Facet By", options=["None", "smoker", "sex", "region"], index=0)
            opacity = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
            
            # Animation option (new feature)
            if chart_type in ["Scatter Plot", "Bubble Chart"]:
                enable_animation = st.checkbox("Enable Animation", value=False)
                if enable_animation:
                    animation_col = st.selectbox("Animate By", options=["age", "bmi", "children"], index=0)
            
            # Color scheme selection
            color_scheme = st.selectbox("Color Scheme", options=[
                "Blues", "Reds", "Greens", "Viridis", "Plasma", "Turbo", 
                "Spectral", "RdYlBu", "YlOrRd"  # New color schemes
            ])
    
    # Export options (new feature)
    export_options = st.expander("Export Options")
    with export_options:
        export_format = st.selectbox("Export Format", ["PNG", "JPEG", "SVG", "PDF", "HTML"], index=0)
        export_width = st.number_input("Width (px)", min_value=800, max_value=3000, value=1200)
        export_height = st.number_input("Height (px)", min_value=600, max_value=2500, value=800)
    
    # Render different chart types
    if chart_type == "Scatter Plot":
        fig = px.scatter(
            df, x=x_axis, y=y_axis, 
            color=color_col, 
            hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
            title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
            height=600,
            opacity=opacity,
            log_y=log_scale if y_axis == "charges" else False,
            log_x=log_scale if x_axis == "charges" else False,
            trendline="ols" if trend_line else None,
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
            color_continuous_scale=color_scheme
        )
        
        # Apply faceting if selected
        if facet_by != "None":
            fig = px.scatter(
                df, x=x_axis, y=y_axis, 
                color=color_col,
                facet_col=facet_by,
                hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()} by {facet_by.capitalize()}",
                height=600,
                opacity=opacity,
                log_y=log_scale if y_axis == "charges" else False,
                log_x=log_scale if x_axis == "charges" else False,
                trendline="ols" if trend_line else None,
                color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                color_continuous_scale=color_scheme
            )
    
    elif chart_type == "Bubble Chart":
        # For bubble chart, we need a size parameter
        size_value = size_col if size_col else "bmi"
        
        fig = px.scatter(
            df, x=x_axis, y=y_axis, 
            color=color_col,
            size=size_value,
            hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
            title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}, Size by {size_value.capitalize()}",
            height=600,
            opacity=opacity,
            log_y=log_scale if y_axis == "charges" else False,
            log_x=log_scale if x_axis == "charges" else False,
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
            color_continuous_scale=color_scheme
        )
    
    elif chart_type == "3D Scatter":
        # For 3D scatter, we need a z-axis
        z_axis = st.selectbox("Z-Axis", options=["charges", "age", "bmi", "children"], index=0)
        
        fig = px.scatter_3d(
            df, x=x_axis, y=y_axis, z=z_axis,
            color=color_col,
            size=size_col if size_col else None,
            hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
            title=f"3D View: {x_axis.capitalize()} vs {y_axis.capitalize()} vs {z_axis.capitalize()}",
            opacity=opacity,
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
            color_continuous_scale=color_scheme
        )
        
        fig.update_layout(height=700)
    
    elif chart_type == "Violin Plot":
        # For violin plot, x should be categorical and y numeric
        if x_axis in ["sex", "smoker", "region"]:
            fig = px.violin(
                df, x=x_axis, y=y_axis,
                color=color_col,
                box=True,
                points="all",
                hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                title=f"Distribution of {y_axis.capitalize()} by {x_axis.capitalize()}",
                color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                color_continuous_scale=color_scheme
            )
        else:
            # If x is numeric, we need to bin it
            if x_axis == "age":
                # Create temporary age groups
                temp_df = df.copy()
                temp_df['age_group'] = pd.cut(
                    temp_df['age'], 
                    bins=[0, 20, 30, 40, 50, 60, 100],
                    labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
                )
                
                fig = px.violin(
                    temp_df, x='age_group', y=y_axis,
                    color=color_col,
                    box=True,
                    points="all",
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by Age Group",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
            elif x_axis == "bmi":
                # Create temporary BMI categories
                temp_df = df.copy()
                temp_df['bmi_category'] = pd.cut(
                    temp_df['bmi'],
                    bins=[0, 18.5, 25, 30, 35, 40, 100],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
                )
                
                fig = px.violin(
                    temp_df, x='bmi_category', y=y_axis,
                    color=color_col,
                    box=True,
                    points="all",
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by BMI Category",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
            else:
                # Use numeric x-axis with fewer points (like children)
                fig = px.violin(
                    df, x=x_axis, y=y_axis,
                    color=color_col,
                    box=True,
                    points="all",
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by {x_axis.capitalize()}",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
    
    elif chart_type == "Box Plot":
        # Similar approach as violin plot for categorical x-axis
        if x_axis in ["sex", "smoker", "region"]:
            fig = px.box(
                df, x=x_axis, y=y_axis,
                color=color_col,
                points="all",
                notched=True,  # Add notches to box plot
                hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                title=f"Distribution of {y_axis.capitalize()} by {x_axis.capitalize()}",
                color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                color_continuous_scale=color_scheme
            )
        else:
            # If x is numeric, we need to bin it
            if x_axis == "age":
                # Create temporary age groups
                temp_df = df.copy()
                temp_df['age_group'] = pd.cut(
                    temp_df['age'], 
                    bins=[0, 20, 30, 40, 50, 60, 100],
                    labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
                )
                
                fig = px.box(
                    temp_df, x='age_group', y=y_axis,
                    color=color_col,
                    points="all",
                    notched=True,
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by Age Group",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
            elif x_axis == "bmi":
                # Create temporary BMI categories
                temp_df = df.copy()
                temp_df['bmi_category'] = pd.cut(
                    temp_df['bmi'],
                    bins=[0, 18.5, 25, 30, 35, 40, 100],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
                )
                
                fig = px.box(
                    temp_df, x='bmi_category', y=y_axis,
                    color=color_col,
                    points="all",
                    notched=True,
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by BMI Category",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
            else:
                # Use numeric x-axis with fewer points (like children)
                fig = px.box(
                    df, x=x_axis, y=y_axis,
                    color=color_col,
                    points="all",
                    notched=True,
                    hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                    title=f"Distribution of {y_axis.capitalize()} by {x_axis.capitalize()}",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
    
    elif chart_type == "Density Contour":
        fig = px.density_contour(
            df, x=x_axis, y=y_axis,
            color=color_col,
            marginal_x="histogram",
            marginal_y="histogram",
            title=f"Density Contour of {y_axis.capitalize()} vs {x_axis.capitalize()}",
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
            color_continuous_scale=color_scheme
        )
    
    # New chart type: Heat Map
    elif chart_type == "Heat Map":
        # For heatmap, we need to aggregate the data
        if x_axis in ["age", "bmi"]:
            # Bin continuous variables
            if x_axis == "age":
                temp_df = df.copy()
                temp_df['age_group'] = pd.cut(
                    temp_df['age'], 
                    bins=[18, 25, 35, 45, 55, 65, 100],
                    labels=['18-25', '25-35', '35-45', '45-55', '55-65', '65+']
                )
                heatmap_df = temp_df.groupby(['age_group', y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'bmi_category']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                # Create a pivot table
                pivot_table = heatmap_df.pivot_table(
                    values='charges', 
                    index='age_group', 
                    columns=y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'bmi_category'
                )
                
                fig = px.imshow(
                    pivot_table,
                    labels=dict(x=y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'BMI Category', 
                                y='Age Group', 
                                color='Average Cost ($)'),
                    title=f"Average Insurance Cost by Age Group and {y_axis.capitalize() if y_axis in ['children', 'sex', 'smoker', 'region'] else 'BMI Category'}",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=color_scheme
                )
                
                # Format text as currency
                fig.update_traces(texttemplate="$%{z:.0f}")
                
            elif x_axis == "bmi":
                temp_df = df.copy()
                temp_df['bmi_category'] = pd.cut(
                    temp_df['bmi'],
                    bins=[0, 18.5, 25, 30, 35, 40, 100],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
                )
                
                heatmap_df = temp_df.groupby(['bmi_category', y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'age_group']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                # Create a pivot table
                pivot_table = heatmap_df.pivot_table(
                    values='charges', 
                    index='bmi_category', 
                    columns=y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'age_group'
                )
                
                fig = px.imshow(
                    pivot_table,
                    labels=dict(x=y_axis if y_axis in ['children', 'sex', 'smoker', 'region'] else 'Age Group', 
                                y='BMI Category', 
                                color='Average Cost ($)'),
                    title=f"Average Insurance Cost by BMI Category and {y_axis.capitalize() if y_axis in ['children', 'sex', 'smoker', 'region'] else 'Age Group'}",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=color_scheme
                )
                
                # Format text as currency
                fig.update_traces(texttemplate="$%{z:.0f}")
                
        else:
            # Use categorical variables directly
            if x_axis in ["sex", "smoker", "region"] and y_axis in ["sex", "smoker", "region", "children"]:
                heatmap_df = df.groupby([x_axis, y_axis]).agg({
                    'charges': 'mean'
                }).reset_index()
                
                # Create a pivot table
                pivot_table = heatmap_df.pivot_table(values='charges', index=x_axis, columns=y_axis)
                
                fig = px.imshow(
                    pivot_table,
                    labels=dict(x=y_axis.capitalize(), y=x_axis.capitalize(), color='Average Cost ($)'),
                    title=f"Average Insurance Cost by {x_axis.capitalize()} and {y_axis.capitalize()}",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=color_scheme
                )
                
                # Format text as currency
                fig.update_traces(texttemplate="$%{z:.0f}")
            else:
                # For other combinations, just show a message
                st.warning("Heat maps work best with categorical variables or binned numeric variables")
                # Show a simple scatter plot instead
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, 
                    color=color_col,
                    title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                    height=600,
                    opacity=opacity,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
    
    # New chart type: Line Plot
    elif chart_type == "Line Plot":
        # For line plots, we need to aggregate data
        if x_axis == "age":
            # Group by age and calculate mean
            line_df = df.groupby('age').agg({
                'charges': 'mean'
            }).reset_index()
            
            fig = px.line(
                line_df, x='age', y='charges',
                title=f"Average Insurance Cost by Age",
                labels={'charges': 'Average Cost ($)', 'age': 'Age'},
                markers=True,
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            # Add smoker vs non-smoker line if color is set to smoker
            if color_col == "smoker":
                smoker_line_df = df.groupby(['age', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    smoker_line_df, x='age', y='charges', color='smoker',
                    title=f"Average Insurance Cost by Age and Smoking Status",
                    labels={'charges': 'Average Cost ($)', 'age': 'Age', 'smoker': 'Smoker'},
                    markers=True,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
            # Add gender lines if color is set to sex
            elif color_col == "sex":
                gender_line_df = df.groupby(['age', 'sex']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    gender_line_df, x='age', y='charges', color='sex',
                    title=f"Average Insurance Cost by Age and Gender",
                    labels={'charges': 'Average Cost ($)', 'age': 'Age', 'sex': 'Gender'},
                    markers=True,
                    color_discrete_map={"male": "#3498db", "female": "#9b59b6"}
                )
            # Add region lines if color is set to region
            elif color_col == "region":
                region_line_df = df.groupby(['age', 'region']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    region_line_df, x='age', y='charges', color='region',
                    title=f"Average Insurance Cost by Age and Region",
                    labels={'charges': 'Average Cost ($)', 'age': 'Age', 'region': 'Region'},
                    markers=True
                )
        
        elif x_axis == "bmi":
            # Group by BMI (rounded) and calculate mean
            temp_df = df.copy()
            temp_df['bmi_rounded'] = round(temp_df['bmi'])
            
            line_df = temp_df.groupby('bmi_rounded').agg({
                'charges': 'mean'
            }).reset_index()
            
            fig = px.line(
                line_df, x='bmi_rounded', y='charges',
                title=f"Average Insurance Cost by BMI",
                labels={'charges': 'Average Cost ($)', 'bmi_rounded': 'BMI'},
                markers=True,
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            # Add smoker vs non-smoker line if color is set to smoker
            if color_col == "smoker":
                smoker_line_df = temp_df.groupby(['bmi_rounded', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    smoker_line_df, x='bmi_rounded', y='charges', color='smoker',
                    title=f"Average Insurance Cost by BMI and Smoking Status",
                    labels={'charges': 'Average Cost ($)', 'bmi_rounded': 'BMI', 'smoker': 'Smoker'},
                    markers=True,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
        
        else:
            # For other x-axes, show a simple line chart
            if x_axis in ["children", "sex", "smoker", "region"]:
                line_df = df.groupby(x_axis).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.line(
                    line_df, x=x_axis, y='charges',
                    title=f"Average Insurance Cost by {x_axis.capitalize()}",
                    labels={'charges': 'Average Cost ($)', x_axis: x_axis.capitalize()},
                    markers=True,
                    color_discrete_sequence=[px.colors.sequential.Plasma[4]]
                )
            else:
                # Default to scatter plot if x-axis doesn't make sense for a line chart
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, 
                    color=color_col,
                    title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                    height=600,
                    opacity=opacity,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme
                )
    
    # New chart type: Area Plot
    elif chart_type == "Area Plot":
        if x_axis == "age":
            # Group by age and calculate mean
            area_df = df.groupby('age').agg({
                'charges': 'mean'
            }).reset_index()
            
            fig = px.area(
                area_df, x='age', y='charges',
                title=f"Average Insurance Cost by Age",
                labels={'charges': 'Average Cost ($)', 'age': 'Age'},
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            # Add smoker vs non-smoker areas if color is set to smoker
            if color_col == "smoker":
                smoker_area_df = df.groupby(['age', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.area(
                    smoker_area_df, x='age', y='charges', color='smoker',
                    title=f"Average Insurance Cost by Age and Smoking Status",
                    labels={'charges': 'Average Cost ($)', 'age': 'Age', 'smoker': 'Smoker'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
        elif x_axis == "bmi":
            # Group by BMI (rounded) and calculate mean
            temp_df = df.copy()
            temp_df['bmi_rounded'] = round(temp_df['bmi'])
            
            area_df = temp_df.groupby('bmi_rounded').agg({
                'charges': 'mean'
            }).reset_index()
            
            fig = px.area(
                area_df, x='bmi_rounded', y='charges',
                title=f"Average Insurance Cost by BMI",
                labels={'charges': 'Average Cost ($)', 'bmi_rounded': 'BMI'},
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            # Add smoker vs non-smoker areas if color is set to smoker
            if color_col == "smoker":
                smoker_area_df = temp_df.groupby(['bmi_rounded', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig = px.area(
                    smoker_area_df, x='bmi_rounded', y='charges', color='smoker',
                    title=f"Average Insurance Cost by BMI and Smoking Status",
                    labels={'charges': 'Average Cost ($)', 'bmi_rounded': 'BMI', 'smoker': 'Smoker'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
        else:
            # Default to scatter plot for other x-axes
            fig = px.scatter(
                df, x=x_axis, y=y_axis, 
                color=color_col,
                title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                height=600,
                opacity=opacity,
                color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                color_continuous_scale=color_scheme
            )
    
    # Apply logarithmic scales if requested
    if chart_type not in ["Heat Map", "Violin Plot", "Box Plot"]:
        if log_scale and y_axis == "charges":
            fig.update_yaxes(type="log")
        if log_scale and x_axis == "charges":
            fig.update_xaxes(type="log")
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights based on chart
    insight_expander = st.expander("Chart Insights")
    with insight_expander:
        if x_axis == "age" and y_axis == "charges":
            st.info("🔍 **Insight**: There's a positive correlation between age and insurance costs. Premiums tend to increase with age as health risks rise.")
            
            # Calculate correlation
            correlation = df[['age', 'charges']].corr().iloc[0, 1]
            st.metric("Age-Cost Correlation", f"{correlation:.2f}")
            
            # Show age group averages
            age_group_avgs = df.groupby('age_group')['charges'].mean().reset_index()
            age_group_avgs['charges'] = age_group_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by age group:")
            st.dataframe(age_group_avgs)
            
        elif x_axis == "bmi" and y_axis == "charges":
            st.info("🔍 **Insight**: BMI shows a moderate positive correlation with insurance costs. Higher BMI values are associated with greater health risks and higher premiums.")
            
            # Calculate correlation
            correlation = df[['bmi', 'charges']].corr().iloc[0, 1]
            st.metric("BMI-Cost Correlation", f"{correlation:.2f}")
            
            # Show BMI category averages
            bmi_cat_avgs = df.groupby('bmi_category')['charges'].mean().reset_index()
            bmi_cat_avgs['charges'] = bmi_cat_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by BMI category:")
            st.dataframe(bmi_cat_avgs)
            
        elif color_col == "smoker" and y_axis == "charges":
            st.warning("🔥 **Key Finding**: Smoking has the strongest impact on insurance costs among all factors. The vertical separation between smokers and non-smokers is substantial at all age and BMI levels.")
            
            # Show smoker vs non-smoker average
            smoker_avgs = df.groupby('smoker')['charges'].mean().reset_index()
            smoker_avgs['charges'] = smoker_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by smoking status:")
            st.dataframe(smoker_avgs)
            
        elif x_axis == "children" and y_axis == "charges":
            st.info("🔍 **Insight**: The number of dependents covered has a modest effect on insurance premiums. There's a slight positive correlation between family size and costs.")
            
            # Calculate correlation
            correlation = df[['children', 'charges']].corr().iloc[0, 1]
            st.metric("Dependents-Cost Correlation", f"{correlation:.2f}")
            
        elif (x_axis == "region" or color_col == "region") and y_axis == "charges":
            st.info("🔍 **Insight**: Geographic region can affect insurance costs due to local healthcare prices, regulations, and population health factors.")
            
            # Show region averages
            region_avgs = df.groupby('region')['charges'].mean().reset_index()
            region_avgs['charges'] = region_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by region:")
            st.dataframe(region_avgs)
            
        elif (x_axis == "sex" or color_col == "sex") and y_axis == "charges":
            st.info("🔍 **Insight**: Gender differences in insurance costs are relatively minor compared to factors like age and smoking status.")
            
            # Show gender averages
            gender_avgs = df.groupby('sex')['charges'].mean().reset_index()
            gender_avgs['charges'] = gender_avgs['charges'].map('${:,.2f}'.format)
            st.write("Average costs by gender:")
            st.dataframe(gender_avgs)
        
        # Add data summary statistics section
        st.subheader("Summary Statistics")
        if y_axis in ["charges", "age", "bmi", "children"]:
            stats_df = df[y_axis].describe().reset_index()
            stats_df.columns = ["Statistic", "Value"]
            
            # Format values for currency if charges
            if y_axis == "charges":
                stats_df["Value"] = stats_df["Value"].map('${:,.2f}'.format)
            else:
                stats_df["Value"] = stats_df["Value"].round(2)
                
            st.dataframe(stats_df)
    
    # Add download options
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Download the visualization or data for your reports")
    
    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            label="📊 Download Chart",
            data="This feature is not available in the example code.",
            file_name=f"insurance_chart_{chart_type}.png",
            mime="image/png",
            disabled=True
        )
    
    with download_cols[1]:
        st.download_button(
            label="📄 Download Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="insurance_data.csv",
            mime="text/csv"
        )
        
    st.markdown("</div>", unsafe_allow_html=True)

# New implementation for Demographic Analysis tab
with viz_tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Demographic Insurance Cost Analysis")
    
    # Create a dashboard of demographic insights
    demo_tabs = st.tabs(["Age Analysis", "Gender Analysis", "BMI Impact", "Family Size", "Combined Factors"])
    
    with demo_tabs[0]:
        st.markdown("### Age and Insurance Costs")
        
        # Create age groups for better analysis - use numeric values for calculations
        df['age_numeric'] = df['age']  # Keep numeric version for calculations
        
        # Create string age groups for visualization
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 20, 30, 40, 50, 60, 100],
            labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
        )
        
        # Age group statistics - use numeric age for calculations
        age_stats = df.groupby('age_group').agg({
            'charges': ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        age_stats.columns = ['Age Group', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
        
        # Format monetary columns
        age_stats['Mean Cost'] = age_stats['Mean Cost'].map('${:,.2f}'.format)
        age_stats['Median Cost'] = age_stats['Median Cost'].map('${:,.2f}'.format)
        age_stats['Std Dev'] = age_stats['Std Dev'].map('${:,.2f}'.format)
        
        # Display age statistics
        st.dataframe(age_stats, use_container_width=True)
        
        # Create age trend visualization
        fig = px.line(
            df.groupby('age_numeric').agg({'charges': 'mean'}).reset_index(),
            x='age_numeric',
            y='charges',
            title='Average Insurance Cost by Age',
            labels={'charges': 'Average Cost ($)', 'age_numeric': 'Age'},
            markers=True
        )
        
        # Add smoker vs non-smoker trend lines
        smoker_age_data = df.groupby(['age_numeric', 'smoker']).agg({'charges': 'mean'}).reset_index()
        
        fig2 = px.line(
            smoker_age_data,
            x='age_numeric',
            y='charges',
            color='smoker',
            title='Average Insurance Cost by Age and Smoking Status',
            labels={'charges': 'Average Cost ($)', 'age_numeric': 'Age'},
            color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"},
            markers=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Add average cost increase per year
        age_increase = st.expander("Age Cost Analysis")
        with age_increase:
            # Calculate avg cost increase per year of age
            age_costs = df.groupby('age_numeric')['charges'].mean().reset_index()
            
            # Calculate the average increase per year
            age_costs['next_age_cost'] = age_costs['charges'].shift(-1)
            age_costs['cost_increase'] = age_costs['next_age_cost'] - age_costs['charges']
            
            avg_increase_per_year = age_costs['cost_increase'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Cost Increase Per Year of Age", f"${avg_increase_per_year:.2f}")
            
            with col2:
                # Calculate the percentage increase from youngest to oldest
                youngest_cost = age_costs.iloc[0]['charges']
                oldest_cost = age_costs.iloc[-2]['charges']  # -2 to avoid NaN in the last row
                pct_increase = ((oldest_cost - youngest_cost) / youngest_cost) * 100
                
                st.metric("Percentage Increase from Youngest to Oldest", f"{pct_increase:.1f}%")
            
            # Show a bar chart of cost increase by decade
            decade_bins = [18, 30, 40, 50, 60, 70, 100]
            decade_labels = ['18-30', '30-40', '40-50', '50-60', '60-70', '70+']
            
            df['decade'] = pd.cut(df['age'], bins=decade_bins, labels=decade_labels)
            decade_costs = df.groupby('decade')['charges'].mean().reset_index()
            
            fig = px.bar(
                decade_costs,
                x='decade',
                y='charges',
                title='Average Insurance Cost by Age Decade',
                labels={'charges': 'Average Cost ($)', 'decade': 'Age Group'},
                text_auto='.0f',
                color_discrete_sequence=[px.colors.sequential.Plasma[4]]
            )
            
            fig.update_traces(texttemplate='$%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Age-related insights
        st.markdown("""
        <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
            <h4 style="margin-top: 0; color: #3498db;">Age Insights</h4>
            <ul>
                <li><strong>Linear Growth:</strong> Insurance costs generally increase linearly with age</li>
                <li><strong>Higher Variance:</strong> Older age groups show greater variability in costs</li>
                <li><strong>Accelerated Increase:</strong> Costs rise more steeply after age 50</li>
                <li><strong>Smoking Amplifier:</strong> The cost increase with age is much steeper for smokers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # The rest of your Demographic Analysis tab code...
    with demo_tabs[1]:
        st.markdown("### Gender Analysis")
        
        # Gender statistics
        gender_stats = df.groupby('sex').agg({
            'charges': ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        gender_stats.columns = ['Gender', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
        
        # Format monetary columns
        gender_stats['Mean Cost'] = gender_stats['Mean Cost'].map('${:,.2f}'.format)
        gender_stats['Median Cost'] = gender_stats['Median Cost'].map('${:,.2f}'.format)
        gender_stats['Std Dev'] = gender_stats['Std Dev'].map('${:,.2f}'.format)
        
        # Gender and age analysis - use age_group string column for display
        gender_age = df.groupby(['sex', 'age_group']).agg({
            'charges': 'mean'
        }).reset_index()
        
        # Create visualizations
        fig1 = px.bar(
            df.groupby('sex').agg({'charges': 'mean'}).reset_index(),
            x='sex',
            y='charges',
            color='sex',
            title='Average Insurance Cost by Gender',
            labels={'charges': 'Average Cost ($)', 'sex': 'Gender'},
            color_discrete_map={"male": "#3498db", "female": "#9b59b6"},
            text_auto='.2f'
        )
        
        fig1.update_traces(texttemplate='$%{text}', textposition='outside')
        
        fig2 = px.bar(
            gender_age,
            x='age_group',
            y='charges',
            color='sex',
            barmode='group',
            title='Average Insurance Cost by Gender and Age Group',
            labels={'charges': 'Average Cost ($)', 'age_group': 'Age Group', 'sex': 'Gender'},
            color_discrete_map={"male": "#3498db", "female": "#9b59b6"}
        )
        
        # Display gender statistics and charts
        st.dataframe(gender_stats, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Enhanced gender analysis
        gender_analysis = st.expander("Detailed Gender Analysis")
        with gender_analysis:
            # Calculate gender disparity by different factors
            st.subheader("Gender Cost Differences by Factor")
            
            # By smoking status
            gender_smoking_df = df.groupby(['sex', 'smoker']).agg({
                'charges': 'mean'
            }).reset_index()
            
            gender_smoking_pivot = gender_smoking_df.pivot(index='smoker', columns='sex', values='charges')
            gender_smoking_pivot['difference'] = gender_smoking_pivot['male'] - gender_smoking_pivot['female']
            gender_smoking_pivot['pct_difference'] = (gender_smoking_pivot['difference'] / gender_smoking_pivot['female']) * 100
            
            gender_smoking_pivot['male'] = gender_smoking_pivot['male'].map('${:,.2f}'.format)
            gender_smoking_pivot['female'] = gender_smoking_pivot['female'].map('${:,.2f}'.format)
            gender_smoking_pivot['difference'] = gender_smoking_pivot['difference'].map('${:,.2f}'.format)
            gender_smoking_pivot['pct_difference'] = gender_smoking_pivot['pct_difference'].round(1).astype(str) + '%'
            
            gender_smoking_pivot = gender_smoking_pivot.reset_index()
            gender_smoking_pivot.columns = ['Smoking Status', 'Male Cost', 'Female Cost', 'Cost Difference', 'Percentage Difference']
            
            st.write("Gender difference by smoking status:")
            st.dataframe(gender_smoking_pivot, use_container_width=True)
            
            # By BMI category
            gender_bmi_df = df.groupby(['sex', 'bmi_category']).agg({
                'charges': 'mean'
            }).reset_index()
            
            fig3 = px.bar(
                gender_bmi_df,
                x='bmi_category',
                y='charges',
                color='sex',
                barmode='group',
                title='Average Insurance Cost by Gender and BMI Category',
                labels={'charges': 'Average Cost ($)', 'bmi_category': 'BMI Category', 'sex': 'Gender'},
                color_discrete_map={"male": "#3498db", "female": "#9b59b6"}
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Gender and smoking status
        gender_smoking = df.groupby(['sex', 'smoker']).agg({
            'charges': ['mean', 'count']
        }).reset_index()
        
        gender_smoking.columns = ['Gender', 'Smoker', 'Average Cost', 'Count']
        gender_smoking['Average Cost'] = gender_smoking['Average Cost'].map('${:,.2f}'.format)
        
        st.subheader("Gender and Smoking Status")
        st.dataframe(gender_smoking, use_container_width=True)
        
        # Gender insights
        st.markdown("""
        <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
            <h4 style="margin-top: 0; color: #3498db;">Gender Insights</h4>
            <ul>
                <li><strong>Small Difference:</strong> Gender alone is a relatively minor factor in insurance costs</li>
                <li><strong>Age Interaction:</strong> Gender differences vary by age group</li>
                <li><strong>Smoking Impact:</strong> Smoking status has a much larger effect than gender</li>
                <li><strong>Combined Factors:</strong> When combined with other risk factors, gender may have more significant effects</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with demo_tabs[2]:
        st.markdown("### BMI Impact Analysis")
        
        # Create BMI categories
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
        )
        
        # BMI statistics
        bmi_stats = df.groupby('bmi_category').agg({
            'charges': ['mean', 'median', 'std', 'count']
        }).reset_index()
        bmi_stats.columns = ['BMI Category', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
        
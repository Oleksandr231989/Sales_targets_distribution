import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt

st.set_page_config(page_title="Regional Sales Goals Calculator", layout="wide")

# Custom CSS for scrollable tabs and improved UI
st.markdown(
    """
    <style>
    /* Make SKU Navigation tabs scrollable */
    div[role="tablist"] {
        overflow-x: auto;
        white-space: nowrap;
        display: flex;
        flex-wrap: nowrap;
    }
    div[role="tablist"] > div {
        display: inline-block;
        flex: 0 0 auto;
    }
    /* Style for better tab visibility */
    .stTabs [role="tab"] {
        padding: 8px 16px;
        margin-right: 4px;
        border-radius: 4px 4px 0 0;
        background-color: #f0f2f6;
        color: #333;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
        color: #ff4b4b;
    }
    /* Improve spacing and button styling */
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 4px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #e04343;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def clean_percentage(value):
    """Convert percentage string to float, handling comma as decimal separator."""
    if isinstance(value, str):
        value = value.replace(',', '.').replace('%', '')
        try:
            return float(value)
        except ValueError:
            return 0
    return float(value)

def calculate_targets_by_sku(df, sku_targets, sku_growth_factors, sku_min_growth_pcts, sku_product_types):
    """
    Calculate sales targets by SKU based on product type.
    Established products use market share index; Launch products use market size weights.
    """
    df_clean = df.copy()
    
    if df_clean['MS'].dtype == 'object':
        df_clean['MS'] = df_clean['MS'].apply(clean_percentage)
    if df_clean['MS'].max() < 1:
        df_clean['MS'] *= 100
    
    results = []
    
    for sku, sku_group in df_clean.groupby('SKU'):
        if sku not in sku_targets:
            continue
            
        total_target_sales = sku_targets[sku]
        product_type = sku_product_types.get(sku, "Established")
        total_current_sales = sku_group['Product sales'].sum()
        
        if product_type == "Established":
            growth_factor = sku_growth_factors.get(sku, 1.0)
            min_growth_pct = sku_min_growth_pcts.get(sku, 0.5)
            
            # Calculate average market share across regions
            avg_market_share = sku_group['MS'].mean()
            
            # Create a copy of the group and calculate market share index
            sku_with_index = sku_group.copy()
            sku_with_index['MS_Index'] = (sku_with_index['MS'] / avg_market_share) * 100
            
            # Sort by market share index (descending)
            sku_sorted = sku_with_index.sort_values('MS_Index', ascending=False).copy()
            
            # Calculate the inverse index (higher for lower market share)
            sku_sorted['Inverse_Index'] = 200 - sku_sorted['MS_Index']
            sku_sorted['Inverse_Index'] = sku_sorted['Inverse_Index'].clip(lower=10)  # Ensure minimum value
            
            # Apply growth factor to the inverse index
            if growth_factor != 1.0:
                sku_sorted['Inverse_Index'] = np.power(sku_sorted['Inverse_Index'] / 100, growth_factor) * 100
            
            # Normalize the inverse index to get proper growth distribution
            overall_growth_pct = (total_target_sales / total_current_sales - 1) * 100
            is_decline = overall_growth_pct < 0
            
            if is_decline:
                max_decline_pct = min(overall_growth_pct * 2, min_growth_pct - 0.1)
                effective_range = min_growth_pct - max_decline_pct
                
                # Normalize inverse index for decline scenario
                total_inverse = sku_sorted['Inverse_Index'].sum()
                sku_sorted['Inverse_Weight'] = sku_sorted['Inverse_Index'] / total_inverse
                
                # Calculate growth percentage based on inverse weight
                sku_sorted['growthPercent'] = min_growth_pct - sku_sorted['Inverse_Weight'] * effective_range * len(sku_sorted)
            else:
                max_growth_pct = max(overall_growth_pct * 2, min_growth_pct + 0.1)
                effective_range = max_growth_pct - min_growth_pct
                
                # Normalize inverse index for growth scenario
                total_inverse = sku_sorted['Inverse_Index'].sum()
                sku_sorted['Inverse_Weight'] = sku_sorted['Inverse_Index'] / total_inverse
                
                # Calculate growth percentage based on inverse weight
                sku_sorted['growthPercent'] = min_growth_pct + sku_sorted['Inverse_Weight'] * effective_range * len(sku_sorted)
            
            # Calculate target sales based on growth percentages
            sku_sorted['targetSales'] = (sku_sorted['Product sales'] * (1 + sku_sorted['growthPercent'] / 100)).round(2)
            sku_sorted['growthAmount'] = (sku_sorted['targetSales'] - sku_sorted['Product sales']).round(2)
            
            # Adjust to ensure total matches the target
            current_total = sku_sorted['targetSales'].sum()
            if current_total != total_target_sales:
                difference = total_target_sales - current_total
                current_growth = current_total - total_current_sales
                if current_growth != 0:
                    adjustment_factor = (total_target_sales - total_current_sales) / current_growth
                    sku_sorted['growthAmount'] = (sku_sorted['growthAmount'] * adjustment_factor).round(2)
                    sku_sorted['targetSales'] = (sku_sorted['Product sales'] + sku_sorted['growthAmount']).round(2)
                    sku_sorted['growthPercent'] = ((sku_sorted['targetSales'] / sku_sorted['Product sales'] - 1) * 100).round(2)
                else:
                    proportion = sku_sorted['Product sales'] / total_current_sales
                    sku_sorted['growthAmount'] = (difference * proportion).round(2)
                    sku_sorted['targetSales'] = (sku_sorted['Product sales'] + sku_sorted['growthAmount']).round(2)
                    sku_sorted['growthPercent'] = (sku_sorted['growthAmount'] / sku_sorted['Product sales'] * 100).round(2).where(sku_sorted['Product sales'] > 0, 0)
            
            # Clean up temporary columns
            results_columns = [col for col in sku_sorted.columns if col not in ['MS_Index', 'Inverse_Index', 'Inverse_Weight']]
            results.append(sku_sorted[results_columns])
        else:
            # Launch products logic
            sku_group_with_targets = sku_group.copy()
            total_market_size = sku_group_with_targets['Market sales'].sum()
            
            if total_market_size > 0:
                sku_group_with_targets['marketWeight'] = (sku_group_with_targets['Market sales'] / total_market_size * 100).round(2)
                sku_group_with_targets['targetSales'] = (sku_group_with_targets['marketWeight'] / 100 * total_target_sales).round(2)
                sku_group_with_targets['growthAmount'] = (sku_group_with_targets['targetSales'] - sku_group_with_targets['Product sales']).round(2)
                sku_group_with_targets['growthPercent'] = (
                    ((sku_group_with_targets['targetSales'] / sku_group_with_targets['Product sales'] - 1) * 100)
                    .where(sku_group_with_targets['Product sales'] > 0, 
                           np.where(sku_group_with_targets['targetSales'] > 0, 100, 0))
                ).round(2)
                
                current_total = sku_group_with_targets['targetSales'].sum()
                if current_total != total_target_sales:
                    idx_largest = sku_group_with_targets['Market sales'].idxmax()
                    sku_group_with_targets.loc[idx_largest, 'targetSales'] += (total_target_sales - current_total)
                    sku_group_with_targets.loc[idx_largest, 'growthAmount'] = (
                        sku_group_with_targets.loc[idx_largest, 'targetSales'] - 
                        sku_group_with_targets.loc[idx_largest, 'Product sales']
                    ).round(2)
                    sku_group_with_targets.loc[idx_largest, 'growthPercent'] = (
                        (sku_group_with_targets.loc[idx_largest, 'targetSales'] / 
                         sku_group_with_targets.loc[idx_largest, 'Product sales'] - 1) * 100
                        if sku_group_with_targets.loc[idx_largest, 'Product sales'] > 0 
                        else (100 if sku_group_with_targets.loc[idx_largest, 'targetSales'] > 0 else 0)
                    ).round(2)
            else:
                n_regions = len(sku_group_with_targets)
                if n_regions > 0:
                    per_region = round(total_target_sales / n_regions, 2)
                    sku_group_with_targets['targetSales'] = per_region
                    sku_group_with_targets.iloc[-1, sku_group_with_targets.columns.get_loc('targetSales')] += (
                        total_target_sales - per_region * n_regions
                    )
                    sku_group_with_targets['growthAmount'] = (
                        sku_group_with_targets['targetSales'] - sku_group_with_targets['Product sales']
                    ).round(2)
                    sku_group_with_targets['growthPercent'] = (
                        ((sku_group_with_targets['targetSales'] / sku_group_with_targets['Product sales'] - 1) * 100)
                        .where(sku_group_with_targets['Product sales'] > 0, 
                               np.where(sku_group_with_targets['targetSales'] > 0, 100, 0))
                    ).round(2)
            
            results.append(sku_group_with_targets)
    
    return pd.concat(results) if results else pd.DataFrame()

def split_targets_by_month(results_df, monthly_split_df):
    """Split target sales into monthly values based on percentages."""
    results_with_monthly = results_df.copy()
    month_columns = [col for col in monthly_split_df.columns if col != 'SKU']
    
    for col in month_columns:
        results_with_monthly[f'Month_{col}'] = 0.0
    
    for idx, row in results_with_monthly.iterrows():
        sku = row['SKU']
        target_sales = row['targetSales']
        sku_split = monthly_split_df[monthly_split_df['SKU'] == sku]
        
        if not sku_split.empty:
            for month in month_columns:
                month_pct = sku_split[month].values[0] / 100
                results_with_monthly.loc[idx, f'Month_{month}'] = round(target_sales * month_pct, 2)
            
            monthly_sum = sum(results_with_monthly.loc[idx, f'Month_{month}'] for month in month_columns)
            if monthly_sum != target_sales:
                results_with_monthly.loc[idx, f'Month_{month_columns[-1]}'] += (target_sales - monthly_sum)
    
    return results_with_monthly

def format_percentage(value):
    """Format number as percentage with comma separator."""
    return f"{value:.2f}".replace('.', ',') + "%"

def get_product_targets_from_sheet(excelFile):
    """Extract product targets from 'product targets' sheet."""
    try:
        targets_df = pd.read_excel(excelFile, sheet_name='product targets')
        required_columns = ['SKU', 'Target']
        if not all(col in targets_df.columns for col in required_columns):
            st.warning("The 'product targets' sheet must have 'SKU' and 'Target' columns.")
            return pd.DataFrame()
        
        targets_df['Product Type'] = targets_df.get('Product Type', 'Established')
        targets_df['Growth Factor'] = targets_df.get('Growth Factor', 1.0)
        targets_df['Min Growth %'] = targets_df.get('Min Growth %', 0.5)
        
        return targets_df
    except Exception as e:
        st.warning(f"Could not read 'product targets' sheet: {e}")
        return pd.DataFrame()

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'data': None,
        'sku_targets': {},
        'sku_growth_factors': {},
        'sku_min_growth_pcts': {},
        'sku_product_types': {},
        'monthly_split_data': None,
        'product_targets_df': None,
        'last_selected_sku': None,
        'selected_skus': None,  # Added for SKU filter persistence
        'results_display': None  # Added to persist calculated results
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    st.title("Regional Sales Goals Calculator")
    st.write("Set regional sales goals per SKU using market share index principles, supporting both growth and decline scenarios.")
    
    initialize_session_state()
    
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload Excel file with region/SKU data",
        type=['xlsx', 'xls'],
        key="main_data_uploader",
        help="Upload an Excel file containing 'sales data' sheet with columns: Region, SKU, Market sales, Product sales, MS, and optionally GR mkt, GR product."
    )
    
    if uploaded_file:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            df = pd.read_excel(uploaded_file, sheet_name="sales data")
            st.session_state.data = df
            st.success("Main data uploaded successfully!")
            
            display_df = df.copy()
            for col in ['MS', 'GR mkt', 'GR product']:
                if col in display_df.columns:
                    if display_df[col].max() < 1:
                        display_df[col] = display_df[col].apply(lambda x: f"{(x*100):.2f}%".replace('.', ','))
                    else:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%".replace('.', ','))
            
            if 'product targets' in excel_file.sheet_names:
                product_targets_df = get_product_targets_from_sheet(uploaded_file)
                if not product_targets_df.empty:
                    st.session_state.product_targets_df = product_targets_df
                    st.success("Product targets loaded from 'product targets' sheet!")
                    for _, row in product_targets_df.iterrows():
                        sku = row['SKU']
                        st.session_state.sku_targets[sku] = row['Target']
                        st.session_state.sku_product_types[sku] = row.get('Product Type', 'Established')
                        st.session_state.sku_growth_factors[sku] = row.get('Growth Factor', 1.0)
                        st.session_state.sku_min_growth_pcts[sku] = row.get('Min Growth %', 0.5)
            
        except Exception as e:
            st.error(f"Error reading main data file: {e}")
            return
    
    st.subheader("Monthly Split (Optional)")
    monthly_split_file = st.file_uploader(
        "Upload monthly distribution file",
        type=['xlsx', 'xls'],
        key="monthly_split_uploader",
        help="Upload an Excel file with SKU and monthly percentage splits (summing to 100% per SKU)."
    )
    
    if monthly_split_file:
        try:
            monthly_split_df = pd.read_excel(monthly_split_file)
            if 'SKU' not in monthly_split_df.columns:
                st.error("Monthly split file must have a 'SKU' column")
            else:
                row_sums = monthly_split_df.drop('SKU', axis=1).sum(axis=1)
                if row_sums.mean() < 2:
                    monthly_split_df.iloc[:, 1:] *= 100
                
                if not all(99 <= x <= 101 for x in row_sums):
                    st.warning("Monthly split percentages normalized to sum to 100%.")
                    for idx, row in monthly_split_df.iterrows():
                        row_sum = sum(row[1:])
                        if row_sum > 0:
                            for col in monthly_split_df.columns[1:]:
                                monthly_split_df.at[idx, col] = (row[col] / row_sum) * 100
                
                st.session_state.monthly_split_data = monthly_split_df
                st.dataframe(monthly_split_df)
                st.success("Monthly split file uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading monthly split file: {e}")
    
    unique_skus = st.session_state.data['SKU'].unique().tolist() if st.session_state.data is not None and 'SKU' in st.session_state.data.columns else []
    if unique_skus:
        st.subheader("SKU Navigation")
        display_skus = [sku if len(sku) <= 20 else sku[:17] + "..." for sku in unique_skus]
        tabs = st.tabs(display_skus)
        
        for i, tab in enumerate(tabs):
            with tab:
                selected_sku = unique_skus[i]
                st.write(f"**Full SKU Name:** {selected_sku}")
                
                show_manual_inputs = st.session_state.product_targets_df is None or st.session_state.product_targets_df.empty
                sku_totals = st.session_state.data.groupby('SKU')['Product sales'].sum()
                
                if selected_sku not in st.session_state.sku_product_types:
                    st.session_state.sku_product_types[selected_sku] = "Established"
                if selected_sku not in st.session_state.sku_growth_factors:
                    st.session_state.sku_growth_factors[selected_sku] = 1.0
                if selected_sku not in st.session_state.sku_min_growth_pcts:
                    st.session_state.sku_min_growth_pcts[selected_sku] = 0.5
                if selected_sku not in st.session_state.sku_targets:
                    st.session_state.sku_targets[selected_sku] = float(sku_totals.get(selected_sku, 0))
                
                current_total = sku_totals.get(selected_sku, 0)
                
                col_sku1, col_sku2 = st.columns(2)
                
                with col_sku1:
                    st.metric("Current Total Product Sales", f"{current_total:,.2f}")
                    if show_manual_inputs:
                        target_total = st.number_input(
                            "Target Total Sales",
                            value=float(st.session_state.sku_targets[selected_sku]),
                            step=1000.0,
                            format="%.2f",
                            key=f"target_{selected_sku}",
                            help="Set the target sales for this SKU."
                        )
                        st.session_state.sku_targets[selected_sku] = target_total
                    else:
                        target_row = st.session_state.product_targets_df[st.session_state.product_targets_df['SKU'] == selected_sku]
                        if not target_row.empty:
                            target_total = target_row.iloc[0]['Target']
                            st.session_state.sku_targets[selected_sku] = target_total
                            st.write(f"**Target Total Sales:** {target_total:,.2f}")
                
                with col_sku2:
                    if show_manual_inputs:
                        product_type = st.selectbox(
                            "Product Type",
                            options=["Established", "Launch"],
                            index=0 if st.session_state.sku_product_types[selected_sku] == "Established" else 1,
                            key=f"product_type_{selected_sku}",
                            help="'Established' uses market share index; 'Launch' uses market size weights."
                        )
                    else:
                        target_row = st.session_state.product_targets_df[st.session_state.product_targets_df['SKU'] == selected_sku]
                        default_product_type = target_row.iloc[0].get('Product Type', st.session_state.sku_product_types[selected_sku]) if not target_row.empty else st.session_state.sku_product_types[selected_sku]
                        product_type = st.selectbox(
                            "Product Type",
                            options=["Established", "Launch"],
                            index=0 if default_product_type == "Established" else 1,
                            key=f"product_type_select_{selected_sku}",
                            help="'Established' uses market share index; 'Launch' uses market size weights."
                        )
                    
                    st.session_state.sku_product_types[selected_sku] = product_type
                
                if product_type == "Established":
                    default_growth_factor = target_row.iloc[0].get('Growth Factor', st.session_state.sku_growth_factors[selected_sku]) if not show_manual_inputs and not target_row.empty else st.session_state.sku_growth_factors[selected_sku]
                    growth_factor = st.slider(
                        "Growth Distribution Factor",
                        min_value=0.5,
                        max_value=5.0,
                        value=float(default_growth_factor),
                        step=0.1,
                        key=f"growth_factor_{selected_sku}",
                        help="Higher values emphasize differences between high/low market share regions."
                    )
                    st.session_state.sku_growth_factors[selected_sku] = growth_factor
                    st.session_state.sku_min_growth_pcts[selected_sku] = 0.5
                else:
                    st.info("Launch products are distributed based on market size proportions.")
                    st.session_state.sku_growth_factors[selected_sku] = 1.0
                    st.session_state.sku_min_growth_pcts[selected_sku] = 0.0
                
                if current_total > 0 and selected_sku in st.session_state.sku_targets:
                    target_total = st.session_state.sku_targets[selected_sku]
                    growth_amount = target_total - current_total
                    growth_percent = (growth_amount / current_total) * 100
                    st.write(f"**Total {'Growth' if growth_amount >= 0 else 'Reduction'} Needed:** {abs(growth_amount):,.2f} ({growth_percent:.2f}%)")
    
    if st.session_state.data is not None:
        st.subheader("Regional Data")
        st.dataframe(display_df if 'display_df' in locals() else st.session_state.data)
        
        if st.button("Calculate Targets", help="Calculate sales targets based on the settings above."):
            if st.session_state.sku_targets:
                with st.spinner("Calculating targets..."):
                    try:
                        results = calculate_targets_by_sku(
                            st.session_state.data,
                            st.session_state.sku_targets,
                            st.session_state.sku_growth_factors,
                            st.session_state.sku_min_growth_pcts,
                            st.session_state.sku_product_types
                        )
                    except Exception as e:
                        st.error(f"Error calculating targets: {e}")
                        return
                    
                    if not results.empty:
                        if st.session_state.monthly_split_data is not None:
                            try:
                                results = split_targets_by_month(results, st.session_state.monthly_split_data)
                            except Exception as e:
                                st.error(f"Error splitting monthly targets: {e}")
                        
                        # Define columns to display, excluding 'marketWeight'
                        display_cols = ['Region', 'SKU', 'Market sales', 'Product sales', 'MS', 'targetSales', 'growthAmount', 'growthPercent']
                        month_cols = [col for col in results.columns if col.startswith('Month_')]
                        if month_cols:
                            display_cols.extend(month_cols)
                        
                        results_display = results[display_cols].copy()
                        
                        # Define column renaming for display
                        column_mapping = {
                            'Region': 'Region',
                            'SKU': 'SKU',
                            'Market sales': 'Market sales',
                            'Product sales': 'Product sales',
                            'MS': 'Market Share (%)',
                            'targetSales': 'Target Sales',
                            'growthAmount': 'Growth Amount',
                            'growthPercent': 'Growth %'
                        }
                        if st.session_state.monthly_split_data is not None:
                            month_columns = [col for col in st.session_state.monthly_split_data.columns if col != 'SKU']
                            for i, month in enumerate(month_columns):
                                column_mapping[f'Month_{month}'] = f'Month {month}'
                        
                        results_display.rename(columns=column_mapping, inplace=True)
                        
                        # Format percentages
                        results_display['Market Share (%)'] = results_display['Market Share (%)'].apply(format_percentage)
                        results_display['Growth %'] = results_display['Growth %'].apply(format_percentage)
                        
                        # Round numeric columns
                        numeric_cols = ['Market sales', 'Product sales', 'Target Sales', 'Growth Amount']
                        numeric_cols.extend([col for col in results_display.columns if col.startswith('Month ')])
                        for col in numeric_cols:
                            if col in results_display.columns:
                                results_display[col] = results_display[col].round(2)
                        
                        # Store results in session state
                        st.session_state.results_display = results_display
                    else:
                        st.warning("No results generated. Check SKU targets.")
                        return
            else:
                st.warning("Set targets for at least one SKU.")
                return

        # Display Calculated Targets table if results exist in session state
        if st.session_state.results_display is not None:
            st.subheader("Calculated Targets")
            st.dataframe(st.session_state.results_display, use_container_width=True)
            
            csv = st.session_state.results_display.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="sales_targets_by_sku.csv",
                mime="text/csv"
            )
            
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    st.session_state.results_display.to_excel(writer, sheet_name='Sales Targets', index=False)
                st.download_button(
                    label="Download Results as Excel",
                    data=buffer.getvalue(),
                    file_name="sales_targets_by_sku.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error creating Excel file: {e}")
                st.info("Install openpyxl: pip install openpyxl")

        # Visualization: Compare Product Sales and Target
        if st.session_state.results_display is not None:
            st.subheader("Sales Comparison by Region")
            
            # SKU Filter
            try:
                sku_options = st.session_state.results_display['SKU'].unique().tolist()
                if not sku_options:
                    st.warning("No SKUs found in the results.")
                    return

                # Initialize session state for selected SKUs if not already set
                if 'selected_skus' not in st.session_state:
                    st.session_state['selected_skus'] = sku_options  # Default to all SKUs

                # Update selected SKUs based on user input
                selected_skus = st.multiselect(
                    "Select SKUs to Display",
                    options=sku_options,
                    default=st.session_state['selected_skus'],  # Use session state for persistence
                    key="sku_filter"
                )

                # Ensure selected_skus is never empty before filtering
                effective_skus = selected_skus if selected_skus else sku_options

                # Update session state with the new selection
                st.session_state['selected_skus'] = effective_skus

                # Filter results based on effective SKUs
                filtered_results = st.session_state.results_display[st.session_state.results_display['SKU'].isin(effective_skus)]

                if filtered_results.empty:
                    st.warning("No data available for the selected SKUs.")
                    return

                # Prepare data for grouped bar chart
                try:
                    # Melt the DataFrame to long format for Altair
                    chart_data = filtered_results.melt(
                        id_vars=['Region'],
                        value_vars=['Product sales', 'Target Sales'],
                        var_name='Sales Type',
                        value_name='Sales'
                    )

                    # Rename 'Target Sales' to 'Target'
                    chart_data['Sales Type'] = chart_data['Sales Type'].replace('Target Sales', 'Target')

                    # Create grouped bar chart with Altair
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X('Region:N', title='Region', axis=alt.Axis(labelAngle=45)),
                        y=alt.Y('Sales:Q', title='Sales'),
                        xOffset='Sales Type:N',  # Group bars by Sales Type
                        color=alt.Color('Sales Type:N', scale=alt.Scale(
                            domain=['Product sales', 'Target'],
                            range=['#1f77b4', '#ff7f0e']
                        ), legend=alt.Legend(title='Sales Type')),
                        tooltip=['Region', 'Sales Type', 'Sales']
                    ).properties(
                        height=600,
                        width='container'
                    ).configure_axis(
                        labelFontSize=12
                    )

                    # Display the chart
                    st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.error(f"Error creating visualization: {e}")

            except Exception as e:
                st.error(f"Error setting up SKU filter: {e}")
    else:
        st.info("Upload an Excel file with region and SKU data to start.")
        st.markdown("""
        ### Expected Format for Main Data
        - **Region**: Region name
        - **SKU**: Product SKU
        - **Market sales**: Market value
        - **Product sales**: Current product sales
        - **GR mkt**: Market growth percentage
        - **GR product**: Product growth percentage
        - **MS**: Market share percentage
        
        ### Expected Format for Product Targets Sheet
        - **SKU**: Matches main data SKUs
        - **Target**: Target sales
        - **Product Type**: 'Established' or 'Launch' (optional)
        - **Growth Factor**: 0.5 to 5.0 for Established (optional)
        - **Min Growth %**: Minimum growth percentage (optional)
        
        ### Expected Format for Monthly Split File
        - **SKU**: Matches main data SKUs
        - **Month columns**: Percentages summing to 100% per SKU
        """)

if __name__ == "__main__":
    main()

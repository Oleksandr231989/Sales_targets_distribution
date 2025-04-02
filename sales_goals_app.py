import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Regional Sales Goals Calculator", layout="wide")

# Function to load product targets from Excel file
def get_product_targets(file):
    try:
        # Try to read the 'product targets' sheet
        targets_df = pd.read_excel(file, sheet_name='product targets')
        
        # Convert to simple dictionary
        targets = {}
        for _, row in targets_df.iterrows():
            product = row['Product']
            plan = float(row['Plan'])  # Convert to float
            targets[product] = plan
            
        return targets
    except:
        return {}

# Main calculation function (simplified version of your original)
def calculate_targets_by_sku(df, sku_targets, sku_growth_factors, sku_min_growth_pcts, sku_product_types):
    # Your existing calculation function
    # Create a clean copy of the dataframe
    df_clean = df.copy()
    
    # Convert MS to float if it's a string or has percentage format
    if df_clean['MS'].dtype == 'O':  # Object type means it might be a string
        df_clean['MS'] = df_clean['MS'].apply(lambda x: float(str(x).replace(',', '.').replace('%', '')) if isinstance(x, str) else x)
    
    # Check if values are too small (likely in decimal format instead of percentage)
    # If the maximum MS value is less than 1, multiply all values by 100
    if df_clean['MS'].max() < 1:
        df_clean['MS'] = df_clean['MS'] * 100
    
    # Process each SKU separately
    results = []
    
    for sku, sku_group in df_clean.groupby('SKU'):
        # Skip if SKU is not in targets
        if sku not in sku_targets:
            continue
            
        # Get SKU-specific parameters
        total_target_sales = sku_targets[sku]
        product_type = sku_product_types.get(sku, "Established")  # Default to Established if not specified
            
        # Calculate total current sales for this SKU
        total_current_sales = sku_group['MAT Product'].sum()
        
        # Different calculation based on product type
        if product_type == "Established":
            # For established products, use inverse market share principle
            growth_factor = sku_growth_factors.get(sku, 1.0)  # Default to 1.0 if not specified
            min_growth_pct = sku_min_growth_pcts.get(sku, 0.5)  # Default to 0.5% if not specified
            
            # Sort regions by market share in descending order (highest first)
            sku_sorted = sku_group.sort_values('MS', ascending=False).copy()
            
            # Calculate the overall growth percentage needed (can be negative for decline)
            overall_growth_pct = (total_target_sales / total_current_sales - 1) * 100
            
            # Determine if this is a growth or decline scenario
            is_decline = overall_growth_pct < 0
            
            # STEP 1: Calculate positions based on market share
            n_regions = len(sku_sorted)
            positions = np.arange(n_regions)
            
            # Apply growth factor to make the distribution more or less aggressive
            if growth_factor != 1.0 and n_regions > 1:
                normalized_positions = positions / (n_regions - 1)  # 0 to 1
                modified_positions = np.power(normalized_positions, growth_factor)
                positions = modified_positions * (n_regions - 1)
            
            # Calculate effective range for growth percentages
            if is_decline:
                # For decline, we still want inverse relationship:
                # - Higher MS (pos 0) gets more negative percentage
                # - Lower MS (pos n-1) gets less negative percentage
                if n_regions > 1:
                    positions = (n_regions - 1) - positions
                
                # Calculate range from min_growth_pct (upper bound) to 2x the overall decline (lower bound)
                max_decline_pct = min(overall_growth_pct * 2, min_growth_pct - 0.1)
                effective_range = min_growth_pct - max_decline_pct
                
                # Distribute percentages (higher MS = more negative growth)
                if n_regions > 1:
                    sku_sorted['growthPercent'] = min_growth_pct - (positions / (n_regions - 1)) * effective_range
                else:
                    sku_sorted['growthPercent'] = overall_growth_pct  # Single region gets overall growth
            else:
                # For growth, the standard approach:
                # - Higher MS (pos 0) gets minimum growth
                # - Lower MS (pos n-1) gets maximum growth
                max_growth_pct = max(overall_growth_pct * 2, min_growth_pct + 0.1)
                effective_range = max_growth_pct - min_growth_pct
                
                # Distribute percentages (lower MS = higher growth)
                if n_regions > 1:
                    sku_sorted['growthPercent'] = min_growth_pct + (positions / (n_regions - 1)) * effective_range
                else:
                    sku_sorted['growthPercent'] = overall_growth_pct  # Single region gets overall growth
            
            # STEP 2: Calculate target sales and growth amounts
            sku_sorted['targetSales'] = round(sku_sorted['MAT Product'] * (1 + sku_sorted['growthPercent'] / 100))
            sku_sorted['growthAmount'] = sku_sorted['targetSales'] - sku_sorted['MAT Product']
            
            # STEP 3: Adjust to match target total
            current_total = sku_sorted['targetSales'].sum()
            
            if current_total != total_target_sales:
                # Calculate adjustment factor to scale all growth amounts proportionally
                difference = total_target_sales - current_total
                current_growth = current_total - total_current_sales
                
                if current_growth != 0:  # Avoid division by zero
                    adjustment_factor = (total_target_sales - total_current_sales) / current_growth
                    
                    # Apply adjustment to maintain proportionality
                    sku_sorted['growthAmount'] = round(sku_sorted['growthAmount'] * adjustment_factor)
                    sku_sorted['targetSales'] = sku_sorted['MAT Product'] + sku_sorted['growthAmount']
                    sku_sorted['growthPercent'] = (sku_sorted['targetSales'] / sku_sorted['MAT Product'] - 1) * 100
                else:
                    # If current growth is zero but we need to match a different target
                    # Distribute the difference proportionally to region sizes
                    for idx in sku_sorted.index:
                        proportion = sku_sorted.loc[idx, 'MAT Product'] / total_current_sales
                        adjustment = round(difference * proportion)
                        sku_sorted.loc[idx, 'growthAmount'] = adjustment
                        sku_sorted.loc[idx, 'targetSales'] = sku_sorted.loc[idx, 'MAT Product'] + adjustment
                        sku_sorted.loc[idx, 'growthPercent'] = (adjustment / sku_sorted.loc[idx, 'MAT Product']) * 100 if sku_sorted.loc[idx, 'MAT Product'] > 0 else 0
            
            # Add to results
            results.append(sku_sorted)
        else:
            # For Launch products, distribute targets based on market size (MAT market)
            sku_group_with_targets = sku_group.copy()
            
            # Calculate total market size for this SKU
            total_market_size = sku_group_with_targets['MAT market'].sum()
            
            if total_market_size > 0:
                # Calculate weights based on market size
                sku_group_with_targets['marketWeight'] = sku_group_with_targets['MAT market'] / total_market_size * 100  # As percentage
                
                # Distribute target sales proportionally to market weights
                sku_group_with_targets['targetSales'] = round(sku_group_with_targets['marketWeight'] / 100 * total_target_sales)
                
                # Calculate growth amounts and percentages
                sku_group_with_targets['growthAmount'] = sku_group_with_targets['targetSales'] - sku_group_with_targets['MAT Product']
                
                # Avoid division by zero for growth percentage calculation
                sku_group_with_targets['growthPercent'] = sku_group_with_targets.apply(
                    lambda x: (x['targetSales'] / x['MAT Product'] - 1) * 100 if x['MAT Product'] > 0 else (100 if x['targetSales'] > 0 else 0), 
                    axis=1
                )
                
                # Adjust to ensure the sum of targetSales equals the total target
                current_total = sku_group_with_targets['targetSales'].sum()
                
                if current_total != total_target_sales:
                    # Find the region with the largest market to adjust
                    idx_largest = sku_group_with_targets['MAT market'].idxmax()
                    sku_group_with_targets.loc[idx_largest, 'targetSales'] += (total_target_sales - current_total)
                    sku_group_with_targets.loc[idx_largest, 'growthAmount'] = sku_group_with_targets.loc[idx_largest, 'targetSales'] - sku_group_with_targets.loc[idx_largest, 'MAT Product']
                    
                    # Recalculate growth percentage for the adjusted region
                    if sku_group_with_targets.loc[idx_largest, 'MAT Product'] > 0:
                        sku_group_with_targets.loc[idx_largest, 'growthPercent'] = (sku_group_with_targets.loc[idx_largest, 'targetSales'] / sku_group_with_targets.loc[idx_largest, 'MAT Product'] - 1) * 100
                    else:
                        sku_group_with_targets.loc[idx_largest, 'growthPercent'] = 100 if sku_group_with_targets.loc[idx_largest, 'targetSales'] > 0 else 0
            else:
                # If total market size is zero, distribute equally
                n_regions = len(sku_group_with_targets)
                if n_regions > 0:
                    per_region = round(total_target_sales / n_regions)
                    sku_group_with_targets['targetSales'] = per_region
                    
                    # Adjust the last region to ensure the sum matches the target
                    last_idx = sku_group_with_targets.index[-1]
                    sku_group_with_targets.loc[last_idx, 'targetSales'] += (total_target_sales - per_region * n_regions)
                    
                    # Calculate growth amounts and percentages
                    sku_group_with_targets['growthAmount'] = sku_group_with_targets['targetSales'] - sku_group_with_targets['MAT Product']
                    sku_group_with_targets['growthPercent'] = sku_group_with_targets.apply(
                        lambda x: (x['targetSales'] / x['MAT Product'] - 1) * 100 if x['MAT Product'] > 0 else (100 if x['targetSales'] > 0 else 0), 
                        axis=1
                    )
            
            # Add to results
            results.append(sku_group_with_targets)
    
    # Combine all SKU results
    if results:
        df_result = pd.concat(results)
        return df_result
    else:
        return pd.DataFrame()  # Empty DataFrame if no results

def format_percentage(value):
    """Format a number as a percentage with comma as decimal separator"""
    return f"{value:.2f}".replace('.', ',') + "%"

def main():
    st.title("Regional Sales Goals Calculator with SKU Support")
    
    # Initialize session state variables for storing data
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'sku_targets' not in st.session_state:
        st.session_state.sku_targets = {}
    if 'sku_growth_factors' not in st.session_state:
        st.session_state.sku_growth_factors = {}
    if 'sku_min_growth_pcts' not in st.session_state:
        st.session_state.sku_min_growth_pcts = {}
    if 'sku_product_types' not in st.session_state:
        st.session_state.sku_product_types = {}
    if 'product_targets_loaded' not in st.session_state:
        st.session_state.product_targets_loaded = False
    
    # Create two columns for the inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Excel File")
        uploaded_file = st.file_uploader("Upload your Excel file with region and SKU data", type=['xlsx', 'xls'])
        
        if uploaded_file:
            try:
                # Read the main data from the "sales data" sheet
                df = pd.read_excel(uploaded_file, sheet_name="sales data")
                st.session_state.data = df
                st.success("Sales data loaded successfully!")
                
                # Load product targets directly and store them
                if not st.session_state.product_targets_loaded:
                    targets = get_product_targets(uploaded_file)
                    if targets:
                        st.session_state.sku_targets = targets
                        st.session_state.product_targets_loaded = True
                        st.success(f"Product targets loaded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display SKU targets and options
    if st.session_state.data is not None:
        with col2:
            st.subheader("Total Figures by SKU")
            
            # Get unique SKUs and calculate totals
            unique_skus = st.session_state.data['SKU'].unique()
            sku_totals = st.session_state.data.groupby('SKU')['MAT Product'].sum()
            
            # Display each SKU's settings
            for sku in unique_skus:
                st.markdown(f"### SKU: {sku}")
                
                # Display current total
                current_total = sku_totals.get(sku, 0)
                st.metric("Current Total Product Sales", f"{current_total:,.0f}")
                
                # Get the target from session state or use current total as default
                target_value = st.session_state.sku_targets.get(sku, float(current_total))
                
                # Create a text input for target with pre-filled value
                st.markdown(f"**Target Total Sales for {sku}**")
                
                # Create a text input that looks like a standard field but is editable
                target_text = st.text_input(
                    f"Enter target value for {sku}",
                    value=str(int(target_value)),  # Convert to integer then string to remove decimals
                    key=f"target_text_{sku}",
                    label_visibility="collapsed"  # Hide the label
                )
                
                # Convert back to float and store
                try:
                    target_float = float(target_text.replace(',', ''))
                    st.session_state.sku_targets[sku] = target_float
                except:
                    st.warning(f"Please enter a valid number for {sku}")
                    st.session_state.sku_targets[sku] = float(current_total)
                
                # Product Type selection
                product_type = st.selectbox(
                    f"Product Type for {sku}",
                    options=["Established", "Launch"],
                    key=f"product_type_{sku}"
                )
                st.session_state.sku_product_types[sku] = product_type
                
                # Only show growth factor for Established products
                if product_type == "Established":
                    growth_factor = st.slider(
                        f"Growth Distribution Factor for {sku}",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                        key=f"growth_factor_{sku}"
                    )
                    st.session_state.sku_growth_factors[sku] = growth_factor
                    
                    min_growth_pct = st.number_input(
                        f"Minimum Growth % per Region for {sku}",
                        min_value=-50.0,
                        max_value=15.0,
                        value=0.5,
                        step=0.5,
                        key=f"min_growth_{sku}"
                    )
                    st.session_state.sku_min_growth_pcts[sku] = min_growth_pct
                else:
                    st.info("Launch products use market size distribution")
                    st.session_state.sku_growth_factors[sku] = 1.0
                    st.session_state.sku_min_growth_pcts[sku] = 0.0
                
                # Calculate and show growth
                try:
                    target_value = float(target_text.replace(',', ''))
                    if current_total > 0:
                        growth_amount = target_value - current_total
                        growth_percent = (growth_amount / current_total) * 100
                        if growth_amount >= 0:
                            st.write(f"Total Growth Needed: {growth_amount:,.0f} ({growth_percent:.2f}%)")
                        else:
                            st.write(f"Total Reduction Needed: {abs(growth_amount):,.0f} ({growth_percent:.2f}%)")
                except:
                    pass
                
                st.markdown("---")
        
        # Show the data
        st.subheader("Regional Data")
        st.dataframe(st.session_state.data)
        
        # Calculate button
        if st.button("Calculate Targets"):
            with st.spinner("Calculating..."):
                results = calculate_targets_by_sku(
                    st.session_state.data,
                    st.session_state.sku_targets,
                    st.session_state.sku_growth_factors,
                    st.session_state.sku_min_growth_pcts,
                    st.session_state.sku_product_types
                )
                
                if not results.empty:
                    st.subheader("Calculated Targets")
                    
                    # Format the results for display
                    display_cols = ['Region', 'SKU', 'MAT market', 'MAT Product', 'MS', 'targetSales', 'growthAmount', 'growthPercent']
                    if 'marketWeight' in results.columns:
                        display_cols.append('marketWeight')
                    
                    results_display = results[display_cols].copy()
                    
                    # Rename columns
                    column_mapping = {
                        'Region': 'Region',
                        'SKU': 'SKU',
                        'MAT market': 'MAT market',
                        'MAT Product': 'MAT Product',
                        'MS': 'Market Share (%)',
                        'targetSales': 'Target Sales',
                        'growthAmount': 'Growth Amount',
                        'growthPercent': 'Growth %',
                        'marketWeight': 'Weight SKU, %'
                    }
                    results_display.rename(columns=column_mapping, inplace=True)
                    
                    # Format percentages
                    results_display['Market Share (%)'] = results_display['Market Share (%)'].apply(format_percentage)
                    results_display['Growth %'] = results_display['Growth %'].apply(format_percentage)
                    if 'Weight SKU, %' in results_display.columns:
                        results_display['Weight SKU, %'] = results_display['Weight SKU, %'].apply(format_percentage)
                    
                    st.dataframe(results_display)
                    
                    # Add download buttons
                    csv = results_display.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sales_targets.csv",
                        mime="text/csv"
                    )
                    
                    # Excel download
                    try:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            results_display.to_excel(writer, sheet_name='Sales Targets', index=False)
                        excel_data = buffer.getvalue()
                        st.download_button(
                            label="Download Results as Excel",
                            data=excel_data,
                            file_name="sales_targets.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Error creating Excel file: {e}")
                else:
                    st.warning("No results generated. Please check your SKU targets.")
    else:
        st.info("Please upload an Excel file to get started.")

if __name__ == "__main__":
    main()

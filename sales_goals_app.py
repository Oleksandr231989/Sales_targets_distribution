"""
Regional Sales Goals Calculator with SKU Support
-------------------------------
This Streamlit app helps allocate sales targets based on strict inverse market share principles.
Supports both growth and decline scenarios with consistent logic.
Calculations are done at the SKU level.

To run:
1. Install required packages: pip install streamlit pandas numpy openpyxl
2. Run: streamlit run this_file.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Regional Sales Goals Calculator with SKU Support", layout="wide")

def clean_percentage(value):
    """Convert percentage string values to float (handling comma as decimal separator)"""
    if isinstance(value, str):
        # Replace comma with dot (for European number format)
        value = value.replace(',', '.')
        # Remove % sign if present
        value = value.replace('%', '')
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            return 0
    return value

def calculate_targets_by_sku(df, sku_targets, sku_growth_factors, sku_min_growth_pcts, sku_product_types):
    """
    Calculate sales targets based on product type and parameters.
    - For Established products: uses inverse market share principle
    - For Launch products: distributes based on market size weight
    
    Parameters:
    - df: DataFrame with region and SKU data including current sales and market share
    - sku_targets: Dict with SKU as key and target sales as value
    - sku_growth_factors: Dict with SKU as key and growth factor as value
    - sku_min_growth_pcts: Dict with SKU as key and minimum growth percentage as value
    - sku_product_types: Dict with SKU as key and product type as value ("Launch" or "Established")
    
    Returns:
    - DataFrame with calculated targets
    """
    # Create a clean copy of the dataframe
    df_clean = df.copy()
    
    # Convert MS to float if it's a string or has percentage format
    if df_clean['MS'].dtype == 'O':  # Object type means it might be a string
        df_clean['MS'] = df_clean['MS'].apply(clean_percentage)
    
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
    
    st.write("""
    This app helps you set sales goals per region and SKU based on strict inverse market share principles. 
    The calculator supports both growth and decline scenarios.
    """)
    
    # Initialize session state variables at the beginning
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
    
    # Create two columns for the inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Excel File")
        uploaded_file = st.file_uploader("Upload your Excel file with region and SKU data", type=['xlsx', 'xls'])
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.success("File successfully uploaded and read!")
                
                # Convert decimal values to percentage format for display
                percent_columns = ['MS', 'GR mkt', 'GR product']
                
                # Create a display copy for showing percentages in the UI
                display_df = df.copy()
                
                for col in percent_columns:
                    if col in display_df.columns:
                        # Check if values are in decimal format (less than 1)
                        if display_df[col].max() < 1:
                            # For display only, format as percentages
                            display_df[col] = display_df[col].apply(lambda x: f"{(x*100):.2f}%".replace('.', ','))
                        else:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%".replace('.', ','))
                
            except Exception as e:
                st.error(f"Error reading the file: {e}")
                return
            
            # Allow user to select the appropriate sheet if multiple sheets exist
            if isinstance(df, dict):
                sheet_name = st.selectbox("Select the sheet with your data:", options=list(df.keys()))
                df = df[sheet_name]
    
    if uploaded_file and 'df' in locals():
        st.session_state.data = df
        
    # Get list of unique SKUs for target settings
    if st.session_state.data is not None and 'SKU' in st.session_state.data.columns:
        unique_skus = st.session_state.data['SKU'].unique()
    else:
        unique_skus = []
        
    with col2:
        st.subheader("Total Figures by SKU")
        
        if st.session_state.data is not None and len(unique_skus) > 0:
            # Calculate and display current totals by SKU
            sku_totals = st.session_state.data.groupby('SKU')['MAT Product'].sum()
            
            for sku in unique_skus:
                current_total = sku_totals.get(sku, 0)
                
                # Create a container for each SKU
                with st.container():
                    st.markdown(f"### SKU: {sku}")
                    st.metric(f"Current Total Product Sales", f"{current_total:,.0f}")
                    
                    # Target input for this SKU
                    target_total = st.number_input(
                        f"Target Total Sales for {sku}",
                        value=float(current_total),
                        step=1000.0,
                        format="%.0f",
                        key=f"target_{sku}"
                    )
                    
                    # Store the target for this SKU
                    st.session_state.sku_targets[sku] = target_total
                    
                    # Product Type selection for this SKU (Launch or Established)
                    product_type = st.selectbox(
                        f"Product Type for {sku}",
                        options=["Established", "Launch"],
                        key=f"product_type_{sku}"
                    )
                    
                    # Store the product type for this SKU
                    st.session_state.sku_product_types[sku] = product_type
                    
                    # Growth Distribution Factor only shown for Established products
                    if product_type == "Established":
                        growth_factor = st.slider(
                            f"Growth Distribution Factor for {sku}",
                            min_value=0.5,
                            max_value=5.0,
                            value=1.0,
                            step=0.1,
                            key=f"growth_factor_{sku}",
                            help="Higher values make the difference between high and low market share regions more pronounced"
                        )
                        
                        # Store the growth factor for this SKU
                        st.session_state.sku_growth_factors[sku] = growth_factor
                        
                        # Minimum Growth Percentage for this SKU
                        min_growth_pct = st.number_input(
                            f"Minimum Growth % per Region for {sku}",
                            min_value=-50.0,
                            max_value=15.0,
                            value=0.5,
                            step=0.5,
                            key=f"min_growth_{sku}",
                            help="Set the minimum growth percentage (or maximum decline if negative) that any region should have"
                        )
                        
                        # Store the minimum growth percentage for this SKU
                        st.session_state.sku_min_growth_pcts[sku] = min_growth_pct
                    else:
                        # For Launch products, show an info message about the distribution method
                        st.info(f"For Launch products, targets will be distributed based on market size (MAT market) proportions.")
                        
                        # Set default values for growth factor and min growth (won't be used but needed for function call)
                        st.session_state.sku_growth_factors[sku] = 1.0
                        st.session_state.sku_min_growth_pcts[sku] = 0.0
                    
                    if current_total > 0:
                        growth_amount = target_total - current_total
                        growth_percent = (growth_amount / current_total) * 100
                        
                        # Display growth or decline info with appropriate phrasing
                        if growth_amount >= 0:
                            st.write(f"Total Growth Needed: {growth_amount:,.0f} ({growth_percent:.2f}%)")
                        else:
                            st.write(f"Total Reduction Needed: {abs(growth_amount):,.0f} ({growth_percent:.2f}%)")
                    
                    st.markdown("---")  # Add a separator between SKUs
            
            # Store product type for each SKU
        else:
            st.info("Please upload a file with SKU data to set targets.")
    
    # Display the data and results
    if st.session_state.data is not None:
        st.subheader("Regional Data")
        if 'display_df' in locals():
            st.dataframe(display_df)
        else:
            st.dataframe(st.session_state.data)
        
        if st.button("Calculate Targets"):
            if len(st.session_state.sku_targets) > 0:
                with st.spinner("Calculating targets..."):
                    # Calculate targets by SKU
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
                        if 'marketWeight' in results.columns:
                            display_cols = ['Region', 'SKU', 'MAT market', 'MAT Product', 'MS', 'targetSales', 'growthAmount', 'growthPercent', 'marketWeight']
                            results_display = results[display_cols].copy()
                            results_display.columns = ['Region', 'SKU', 'MAT market', 'MAT Product', 'Market Share (%)', 'Target Sales', 'Growth Amount', 'Growth %', 'Weight SKU, %']
                            
                            # Format percentages with comma as decimal separator
                            results_display['Market Share (%)'] = results_display['Market Share (%)'].apply(format_percentage)
                            results_display['Growth %'] = results_display['Growth %'].apply(format_percentage)
                            results_display['Weight SKU, %'] = results_display['Weight SKU, %'].apply(format_percentage)
                        else:
                            display_cols = ['Region', 'SKU', 'MAT market', 'MAT Product', 'MS', 'targetSales', 'growthAmount', 'growthPercent']
                            results_display = results[display_cols].copy()
                            results_display.columns = ['Region', 'SKU', 'MAT market', 'MAT Product', 'Market Share (%)', 'Target Sales', 'Growth Amount', 'Growth %']
                            
                            # Format percentages with comma as decimal separator
                            results_display['Market Share (%)'] = results_display['Market Share (%)'].apply(format_percentage)
                            results_display['Growth %'] = results_display['Growth %'].apply(format_percentage)
                        
                        # Format numeric columns as integers (no decimals)
                        integer_cols = ['MAT market', 'MAT Product', 'Target Sales', 'Growth Amount']
                        for col in integer_cols:
                            results_display[col] = results_display[col].astype(int)
                        
                        st.dataframe(results_display, use_container_width=True)
                        
                        # Add download button for the results
                        csv = results_display.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sales_targets_by_sku.csv",
                            mime="text/csv"
                        )
                        
                        # Add an Excel download option
                        try:
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                results_display.to_excel(writer, sheet_name='Sales Targets', index=False)
                            
                            excel_data = buffer.getvalue()
                            st.download_button(
                                label="Download Results as Excel",
                                data=excel_data,
                                file_name="sales_targets_by_sku.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.error(f"Error creating Excel file: {e}")
                            st.info("Please install openpyxl package with: pip install openpyxl")
                    else:
                        st.warning("No results were generated. Please check your SKU targets.")
            else:
                st.warning("Please set targets for at least one SKU before calculating.")
    else:
        st.info("Please upload an Excel file with your region and SKU data to get started.")
        st.markdown("""
        ### Expected Format
        Your Excel file should have the following columns:
        - **Region**: Region name
        - **SKU**: Product SKU 
        - **MAT market**: Market value for the region
        - **MAT Product**: Product sales for the region (current year)
        - **MAT Market N-1**: Market value for the previous year
        - **MAT Product N-1**: Product sales for the previous year
        - **MS**: Market share percentage
        - **GR mkt**: Market growth percentage
        - **GR product**: Product growth percentage
        
        *The app will use the MAT Product column as current sales and MS column for calculations.*
        """)

if __name__ == "__main__":
    main()
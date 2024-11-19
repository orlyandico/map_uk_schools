import os
import requests
import pandas as pd
import folium
from folium.features import DivIcon
import boto3
from botocore.exceptions import ClientError

def extract_lower_age(age_range):
    try:
        return int(age_range.split('-')[0])
    except (ValueError, AttributeError, IndexError):
        return None

def point_score_to_grade(point_score):
    if pd.isna(point_score) or point_score == 0:
        return 'N/A'
    elif point_score >= 50:
        return 'A*'
    elif point_score >= 40:
        return 'A'
    else:
        return '≤B'

def geocode_address(address, index_name="your-place-index-name", region_name="eu-west-2"):
    try:
        location_client = boto3.client('location', region_name=region_name)
        response = location_client.search_place_index_for_text(
            IndexName=index_name,
            Text=address,
            MaxResults=1
        )
        if response['Results']:
            coordinates = response['Results'][0]['Place']['Geometry']['Point']
            return pd.Series({
                'Latitude': coordinates[1],
                'Longitude': coordinates[0]
            })
        return pd.Series({'Latitude': None, 'Longitude': None})
    except ClientError as e:
        print(f"AWS Error geocoding {address}: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
        return pd.Series({'Latitude': None, 'Longitude': None})
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        return pd.Series({'Latitude': None, 'Longitude': None})

def get_grade_color(score, is_independent):
    if is_independent:
        if score >= 50:  # A*
            return '#FF0000'
        elif score >= 40:  # A
            return '#FFA500'
        else:  # ≤B
            return '#FFD700'
    else:
        if score >= 50:  # A*
            return '#000080'
        elif score >= 40:  # A
            return '#0000FF'
        else:  # ≤B
            return '#4169E1'

def get_text_color(background_color):
    bg_color = background_color.lstrip('#')
    rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
    brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
    return '#000000' if brightness > 128 else '#FFFFFF'

def main():
    csv_filename = "2022-2023_england_ks5final.csv"
    df = pd.read_csv(csv_filename, low_memory=False, dtype={'TELNUM': str})
    print(f"Number of raw data rows: {len(df)}")

    columns_to_keep = ['SCHNAME', 'ADDRESS1', 'TOWN', 'PCODE', 'TELNUM', 'ADMPOL_PT', 'GEND1618', 
                      'AGERANGE', 'TPUP1618', 'TALLPPE_ALEV_1618', 'TB3PTSE']
    df_selected = df[columns_to_keep].copy()

    df_selected['TB3PTSE'] = pd.to_numeric(df_selected['TB3PTSE'], errors='coerce').fillna(0)
    df_selected['TALLPPE_ALEV_1618'] = pd.to_numeric(df_selected['TALLPPE_ALEV_1618'], errors='coerce').fillna(0)
    df_selected['Rank'] = df_selected['TB3PTSE'].rank(method='min', ascending=False)

    df_selected = df_selected[df_selected['TB3PTSE'] >= df_selected['TB3PTSE'].quantile(0.90)]
    print(f"Number of schools in P90: {len(df_selected)}")

    df_selected['TB3PTSE_Grade'] = df_selected['TB3PTSE'].apply(point_score_to_grade)
    df_selected['TALLPPE_ALEV_1618_Grade'] = df_selected['TALLPPE_ALEV_1618'].apply(point_score_to_grade)

    latitudes = []
    longitudes = []
    for _, row in df_selected.iterrows():
        address = f"{row['ADDRESS1']}, {row['TOWN']}, {row['PCODE']}"
        lat, lon = geocode_address(address)
        print(f"Processed: {address}")
        latitudes.append(lat)
        longitudes.append(lon)

    df_selected['Latitude'] = latitudes
    df_selected['Longitude'] = longitudes
    df_selected = df_selected.dropna(subset=['Latitude', 'Longitude'])

    output_csv = "processed_school_data.csv"
    df_selected.sort_values(by='TB3PTSE', ascending=False).to_csv(output_csv, index=False)
    print(f"Saved processed data to {output_csv}")

    m = folium.Map(location=[51.5055, -0.0754], zoom_start=10)

    title_html = '''
                <h3 align="center" style="font-size:20px">
                <a href="https://www.compare-school-performance.service.gov.uk/download-data?download=true&regions=0&filters=KS5&fileformat=csv&year=2022-2023&meta=false">Schools by A-Level Performance, 2022-2023 K5 Data</a>
                </h3>
                '''
    m.get_root().html.add_child(folium.Element(title_html))

    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                border-radius: 5px;
                ">
        <div style="font-weight: bold; margin-bottom: 10px;">Independent Schools</div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width: 12px; height: 12px; background-color: #FF0000; margin-right: 5px;"></span>A* (≥50)
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width: 12px; height: 12px; background-color: #FFA500; margin-right: 5px;"></span>A (40-49)
        </div>
        <div style="margin-bottom: 15px;">
            <span style="display:inline-block; width: 12px; height: 12px; background-color: #FFD700; margin-right: 5px;"></span>≤B (<40)
        </div>
        <div style="font-weight: bold; margin-bottom: 10px;">State Schools</div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width: 12px; height: 12px; background-color: #000080; margin-right: 5px;"></span>A* (≥50)
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width: 12px; height: 12px; background-color: #0000FF; margin-right: 5px;"></span>A (40-49)
        </div>
        <div>
            <span style="display:inline-block; width: 12px; height: 12px; background-color: #4169E1; margin-right: 5px;"></span>≤B (<40)
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in df_selected.iterrows():
        is_independent = pd.isna(row['ADMPOL_PT']) or row['ADMPOL_PT'].strip() == ''
        colour = get_grade_color(row['TB3PTSE'], is_independent)
        fontcolour = get_text_color(colour)
        admpol = 'independent' if is_independent else row['ADMPOL_PT']

        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            address_parts = [str(row[field]) for field in ['ADDRESS1', 'TOWN', 'PCODE'] 
                           if pd.notna(row[field]) and str(row[field]).strip()]
            address_string = ', '.join(address_parts)

            popup_html = f"""
            <b>{row['SCHNAME']}</b><br>
            {address_string}<br>
            Phone: {row['TELNUM']}<br>
            Admissions Policy: {admpol}<br>
            Age Range: {row['AGERANGE']}<br>
            Gender: {row['GEND1618']}<br>
            Students (16-18): {row['TPUP1618']}<br>
            Avg best 3 A-levels (TB3PTSE): {row['TB3PTSE']:.2f} (<b>{row['TB3PTSE_Grade']}</b>)<br>
            Avg per A-level (TALLPPE_ALEV_1618): {row['TALLPPE_ALEV_1618']:.2f} (<b>{row['TALLPPE_ALEV_1618_Grade']}</b>)
            """

            icon = DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 30),
                html=f'''
                    <div style="width: 30px; height: 30px;">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 0C7.58 0 4 3.58 4 8c0 5.76 8 16 8 16s8-10.24 8-16c0-4.42-3.58-8-8-8z"
                                fill="{colour}"
                                stroke="black"
                                stroke-width="1"/>
                            <text x="12" y="9" font-family="Arial" font-size="8" font-weight="bold" 
                                  fill="{fontcolour}" text-anchor="middle" dy=".3em">{round(row['TB3PTSE'])}</text>
                        </svg>
                    </div>
                '''
            )
            
            marker = folium.Marker(
                [row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=icon
            )
            marker.add_to(m)

    m.save("schools_map.html")
    print("Map saved as schools_map.html")

if __name__ == "__main__":
    main()
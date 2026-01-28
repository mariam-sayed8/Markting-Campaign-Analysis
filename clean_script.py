import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

class SocialMediaDataCleaner:


  def __init__(self, file_path: str = None, df: pd.DataFrame = None):

    if file_path:
      self.df = pd.read_csv(file_path)
    elif df is not None:
      self.df = df.copy()
    else:
      raise ValueError("Either file_path or df must be provided")


    self.campaign_id_counter = 1
    self.campaign_id_prefix = "CAMP"

    self.company_mapping = self._generate_company_mapping()
    self.cleaned_df = None


    self.interest_mapping = {
      'Health': 'health',
      'Home': 'home',
      'Technology': 'technology',
      'Food': 'food',
      'Fashion': 'fashion'
    }


    self.default_status = 'completed'
    self.default_currency = 'USD'
    self.default_interest = 'other'
    self.default_language = 'English'
    self.default_location = 'Unknown'
    self.default_age_group = 'unknown'
    self.default_gender = 'unknown'
    self.default_objective = 'general'
    self.default_platform = 'unknown'

  def _generate_company_mapping(self) -> Dict[str, Dict[str, Any]]:

    companies = self.df['Company'].unique()
    return {
      company: {
        'Company_ID': f"CMP{idx:05d}",
        'Company_Name': company
      }
      for idx, company in enumerate(sorted(companies), 1)
    }

  def generate_campaign_id(self) -> str:

    campaign_id = f"{self.campaign_id_prefix}_{self.campaign_id_counter:06d}"
    self.campaign_id_counter += 1
    return campaign_id

  @staticmethod
  def normalize_text(text: str) -> str:

    if pd.isna(text):
      return ""

    text = str(text).strip()
    # Remove special characters except spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

  def generate_campaign_name(self, company: str, campaign_goal: str, channel: str) -> str:

    company_norm = self.normalize_text(company)
    goal_norm = self.normalize_text(campaign_goal)
    channel_norm = self.normalize_text(channel)


    company_short = company_norm.split()[0] if ' ' in company_norm else company_norm[:15]


    goal_clean = re.sub(r'[^a-z0-9\s]', '', goal_norm)
    goal_short = goal_clean.replace(' ', '_')[:15]


    channel_clean = channel_norm.capitalize()

    return f"{company_short}_{goal_short}_{channel_clean}"

  def extract_duration_days(self, duration_str: str) -> int:

    if pd.isna(duration_str):
      return 0

    duration_str = str(duration_str).strip()

    # Extract number from string (e.g., "15 Days" -> 15)
    match = re.search(r'(\d+)', duration_str)
    if match:
      try:
        return int(match.group(1))
      except:
        return 0
    return 0

  def calculate_end_date(self, start_date_str: str, duration_days: int) -> str:

    try:

      if isinstance(start_date_str, str):

        try:
          start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:

          try:
            start_date = datetime.strptime(start_date_str, '%m/%d/%Y')
          except ValueError:
            return "unknown"
      else:
        return "unknown"

      end_date = start_date + timedelta(days=duration_days)
      return end_date.strftime('%m/%d/%Y')
    except Exception:
      return "unknown"

  def get_age_group(self, target_audience: str) -> str:

    if pd.isna(target_audience):
      return self.default_age_group

    audience = str(target_audience).strip()


    patterns = {
      '18-24': r'18[-_]?24|18\s*to\s*24',
      '25-34': r'25[-_]?34|25\s*to\s*34',
      '35-44': r'35[-_]?44|35\s*to\s*34',
      '45-60': r'45[-_]?60|45\s*to\s*60',
      'All Ages': r'all\s*ages|all\s*ages',
    }

    for age_group, pattern in patterns.items():
      if re.search(pattern, audience, re.IGNORECASE):
        return age_group

    return self.default_age_group

  def get_gender(self, target_audience: str) -> str:

    if pd.isna(target_audience):
      return self.default_gender

    audience = str(target_audience).lower()

    if 'women' in audience:
      return "Female"
    elif 'men' in audience:
      return "Male"
    elif 'all ages' in audience:
      return "All"
    else:
      return self.default_gender

  def get_interest(self, customer_segment: str) -> str:

    if pd.isna(customer_segment):
      return self.default_interest

    segment = str(customer_segment).strip()
    return self.interest_mapping.get(segment, self.default_interest)

  def get_status(self, end_date_str: str) -> str:

    if end_date_str == "unknown":
      return self.default_status

    try:
      # Parse end date
      end_date = datetime.strptime(end_date_str, '%m/%d/%Y')
      today = datetime.now()

      if end_date < today:
        return "completed"
      elif end_date > today:
        return "active"
      else:
        return "ending_today"
    except:
      return self.default_status

  def get_currency(self) -> str:
    return self.default_currency

  def get_language(self, language_str: str) -> str:
    if pd.isna(language_str):
      return self.default_language

    language = str(language_str).strip()
    return language if language else self.default_language

  def get_location(self, location_str: str) -> str:

    if pd.isna(location_str):
      return self.default_location

    location = str(location_str).strip()
    return location if location else self.default_location

  def get_objective(self, campaign_goal: str) -> str:

    if pd.isna(campaign_goal):
      return self.default_objective

    objective = str(campaign_goal).strip()
    return objective if objective else self.default_objective

  def get_platform(self, channel: str) -> str:

    if pd.isna(channel):
      return self.default_platform

    platform = str(channel).strip()
    return platform if platform else self.default_platform

  def clean_acquisition_cost(self, cost_str: str) -> float:

    if pd.isna(cost_str):
      return 0.0

    cost_str = str(cost_str).strip()

    # Remove currency symbols and commas
    cost_clean = re.sub(r'[^\d.]', '', cost_str)

    try:
      return float(cost_clean)
    except:
      return 0.0

  def calculate_total_budget(self, acquisition_cost: float) -> float:

    if acquisition_cost <= 0:
      return 0.0

    return acquisition_cost

  def calculate_expected_budget(self, acquisition_cost: float) -> float:

    if acquisition_cost <= 0:
      return 0.0

    return acquisition_cost * 1.2

  def transform_data(self) -> pd.DataFrame:

    dfr = self.df.copy()

    dfr['Acquisition_Cost_Clean'] = dfr['Acquisition_Cost'].apply(self.clean_acquisition_cost)

    dfr['cpc'] = dfr['Acquisition_Cost_Clean'] / dfr['Clicks'].replace(0, 1)
    dfr['conversions'] = dfr['Clicks'] * dfr['Conversion_Rate']
    dfr['cpa'] = dfr['Acquisition_Cost_Clean'] / dfr['conversions'].replace(0, 1)
    dfr['ctr'] = (dfr['Clicks'] / dfr['Impressions'].replace(0, 1)) * 100
    dfr['revenue'] = (dfr['ROI'] * dfr['Acquisition_Cost_Clean']) + dfr['Acquisition_Cost_Clean']


    dfr['duration_days'] = dfr['Duration'].apply(self.extract_duration_days)
    dfr['Age_Group'] = dfr['Target_Audience'].apply(self.get_age_group)
    dfr['Gender'] = dfr['Target_Audience'].apply(self.get_gender)
    dfr['interest'] = dfr['Customer_Segment'].apply(self.get_interest)


    dfr['Campaign_Name'] = dfr.apply(
      lambda row: self.generate_campaign_name(
        row['Company'],
        row['Campaign_Goal'],
        row['Channel_Used']
      ),
      axis=1
    )


    dfr['End_Date'] = dfr.apply(
      lambda row: self.calculate_end_date(row['Date'], row['duration_days']),
      axis=1
    )


    dfr['Start_Date'] = dfr['Date'].apply(
      lambda x: datetime.strptime(str(x), '%Y-%m-%d').strftime('%m/%d/%Y')
      if not pd.isna(x) else "unknown"
    )


    dfr['status'] = dfr['End_Date'].apply(self.get_status)
    dfr['Currency'] = dfr.apply(lambda x: self.get_currency(), axis=1)
    dfr['Language'] = dfr['Language'].apply(self.get_language)
    dfr['Location'] = dfr['Location'].apply(self.get_location)
    dfr['Objective'] = dfr['Campaign_Goal'].apply(self.get_objective)
    dfr['Platform_Name'] = dfr['Channel_Used'].apply(self.get_platform)


    dfr['Total_Budget'] = dfr['Acquisition_Cost_Clean'].apply(self.calculate_total_budget)
    dfr['Expected_Budget'] = dfr['Acquisition_Cost_Clean'].apply(self.calculate_expected_budget)


    dfr['Company_ID'] = dfr['Company'].apply(
      lambda x: self.company_mapping.get(x, {}).get('Company_ID', 'unknown')
    )


    campaign_ids = []
    for i in range(len(dfr)):
      campaign_ids.append(self.generate_campaign_id())


    transformed_df = pd.DataFrame({
      'campaign_ID': campaign_ids,
      'Campaign_Name': dfr['Campaign_Name'],
      'Platform_Name': dfr['Platform_Name'],
      'Start_Date': dfr['Start_Date'],
      'End_Date': dfr['End_Date'],
      'interest': dfr['interest'],
      'Total_Budget': round(dfr['Total_Budget'],3),
      'Budget_Spent': round(dfr['Acquisition_Cost_Clean'],3),
      'Expected_Budget': round(dfr['Expected_Budget'],3),
      'Impressions': dfr['Impressions'],
      'Clicks': dfr['Clicks'],
      'Conversions': dfr['conversions'],
      'Revenue': round(dfr['revenue'],3),
      'CTR': round(dfr['ctr'],3),
      'CPC': round(dfr['cpc'],3),
      'CPA': round(dfr['cpa'],3),
      'Conversion_Rate': round(dfr['Conversion_Rate'] * 100,3),
      'engagement_score': dfr['Engagement_Score'],
      'ROI': round(dfr['ROI'],3),
      'Location': dfr['Location'],
      'Age_Group': dfr['Age_Group'],
      'duration_days': dfr['duration_days'],
      'Gender': dfr['Gender'],
      'Objective': dfr['Objective'],
      'Language': dfr['Language'],
      'Company_ID': dfr['Company_ID'],
      'Company_Name': dfr['Company'],
      'status': dfr['status'],
      'Currency': dfr['Currency']
    })

    self.cleaned_df = transformed_df
    return transformed_df

  def add_new_campaigns(self, new_data: pd.DataFrame) -> pd.DataFrame:

    if self.cleaned_df is None:
      self.transform_data()


    current_data = self.cleaned_df.copy()


    new_cleaner = SocialMediaDataCleaner(df=new_data)
    new_campaigns = new_cleaner.transform_data()


    last_campaign_id = int(current_data['campaign_ID'].str.split('_').str[-1].astype(int).max())
    new_cleaner.campaign_id_counter = last_campaign_id + 1


    new_campaign_ids = []
    for i in range(len(new_campaigns)):
      new_campaign_ids.append(new_cleaner.generate_campaign_id())

    new_campaigns['campaign_ID'] = new_campaign_ids


    combined_data = pd.concat([current_data, new_campaigns], ignore_index=True)


    self.cleaned_df = combined_data
    self.campaign_id_counter = new_cleaner.campaign_id_counter

    return combined_data

  def save_cleaned_data(self, output_path: str):

    if self.cleaned_df is None:
      self.transform_data()

    self.cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

  def get_summary(self) -> Dict[str, Any]:
    if self.cleaned_df is None:
      self.transform_data()

    return {
      'original_rows': len(self.df),
      'cleaned_rows': len(self.cleaned_df),
      'companies_count': len(self.company_mapping),
      'unique_campaigns': len(self.cleaned_df['campaign_ID'].unique()),
      'next_campaign_id': f"{self.campaign_id_prefix}_{self.campaign_id_counter:06d}",
      'columns_transformed': len(self.cleaned_df.columns),
      'status_distribution': self.cleaned_df['status'].value_counts().to_dict(),
      'interest_distribution': self.cleaned_df['interest'].value_counts().to_dict(),
      'currency_distribution': self.cleaned_df['Currency'].value_counts().to_dict(),
      'default_values_used': {
        'status': self.default_status,
        'currency': self.default_currency,
        'interest': self.default_interest,
        'language': self.default_language,
        'location': self.default_location,
        'age_group': self.default_age_group,
        'gender': self.default_gender,
        'objective': self.default_objective,
        'platform': self.default_platform
      }
    }


def main():



  input_file = 'Social_Media_Advertising.csv'
  output_file = 'Cleaned_Social_Media_Advertising.csv'

  try:

    cleaner = SocialMediaDataCleaner(file_path=input_file)


    cleaned_data = cleaner.transform_data()


    print("Sample of cleaned data:")
    print(cleaned_data[['campaign_ID', 'Campaign_Name', 'Platform_Name', 'status', 'Currency']].head())


    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    print(f"Original rows: {summary['original_rows']}")
    print(f"Cleaned rows: {summary['cleaned_rows']}")
    print(f"Companies: {summary['companies_count']}")
    print(f"Next campaign ID available: {summary['next_campaign_id']}")

    print("\nStatus Distribution:")
    for status, count in summary['status_distribution'].items():
      print(f"  {status}: {count} campaigns")

    print("\nDefault Values Used:")
    for key, value in summary['default_values_used'].items():
      print(f"  {key}: {value}")


    cleaner.save_cleaned_data(output_file)


    print("\nColumn Mapping:")
    original_cols = list(cleaner.df.columns)
    new_cols = list(cleaned_data.columns)
    print(f"Original columns ({len(original_cols)}):")
    print(f"  {', '.join(original_cols)}")
    print(f"New columns ({len(new_cols)}):")
    print(f"  {', '.join(new_cols)}")


    print("\nKey Metrics:")
    print(f"Total Budget: ${cleaned_data['Total_Budget'].sum():,.2f}")
    print(f"Total Budget Spent: ${cleaned_data['Budget_Spent'].sum():,.2f}")
    print(f"Total Revenue: ${cleaned_data['Revenue'].sum():,.2f}")
    print(f"Average ROI: {cleaned_data['ROI'].mean():.2f}")
    print(f"Average Conversion Rate: {cleaned_data['Conversion_Rate'].mean():.2f}%")
    print(f"First Campaign ID: {cleaned_data['campaign_ID'].iloc[0]}")
    print(f"Last Campaign ID: {cleaned_data['campaign_ID'].iloc[-1]}")

  except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
  except Exception as e:
    print(f"Error during cleaning: {str(e)}")


if __name__ == "__main__":
  main()
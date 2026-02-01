import pandas as pd
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Set


class SocialMediaDataCleaner:

  def __init__(self, file_path: str = None, df: pd.DataFrame = None):

    if file_path:
      self.df = pd.read_csv(file_path)
    elif df is not None:
      self.df = df.copy()
    else:
      raise ValueError("Either file_path or df must be provided")

    self.company_mapping = {}
    self.campaign_mapping = {}
    self.cleaned_df = None

    self.default_values = {
      'status': 'completed',
      'interest': 'other',
      'language': 'unknown',
      'location': 'Unknown',
      'age_group': 'unknown',
      'gender': 'unknown',
      'objective': 'general',
      'platform': 'unknown',
      'company': 'Unknown Company',
      'company_id': 'CMP000001',
      'campaign_id': 'CAMP000001',
      'campaign_goal': 'general',
      'channel': 'unknown',
      'duration': 15,
      'conversion_rate': 0.0,
      'acquisition_cost': 0.0,
      'roi': 0.0,
      'impressions': 0,
      'clicks': 0,
      'engagement_score': 0,
      'start_date': 'unknown',
      'end_date': 'unknown'
    }


    self._init_category_mappings()

    self._init_quick_lookups()

  def _init_category_mappings(self):


    self.interest_categories = {
      'health': ['health', 'wellness', 'fitness', 'nutrition', 'medical'],
      'technology': ['tech', 'software', 'hardware', 'gadgets', 'electronics', 'digital'],
      'fashion': ['fashion', 'clothing', 'apparel', 'style', 'wear'],
      'food': ['food', 'restaurant', 'cuisine', 'dining', 'beverage', 'restaurants'],
      'home': ['home', 'furniture', 'decor', 'household', 'interior'],
      'travel': ['travel', 'tourism', 'vacation', 'hotel', 'flight'],
      'finance': ['finance', 'banking', 'investment', 'money', 'financial'],
      'education': ['education', 'learning', 'course', 'training', 'school'],
      'entertainment': ['entertainment', 'movie', 'music', 'game', 'gaming'],
      'sports': ['sports', 'fitness', 'athletic', 'outdoor']
    }


    self.platform_mapping = {
      'facebook': ['facebook', 'fb', 'facebook ads'],
      'instagram': ['instagram', 'ig', 'insta'],
      'twitter': ['twitter', 'tweet', 'x'],
      'linkedin': ['linkedin', 'linkedin ads'],
      'youtube': ['youtube', 'yt', 'youtube ads'],
      'tiktok': ['tiktok', 'tiktok ads'],
      'google': ['google', 'google ads', 'adwords']
    }


    self.language_mapping = {
      'english': ['english', 'en', 'eng'],
      'spanish': ['spanish', 'es', 'esp'],
      'french': ['french', 'fr', 'français'],
      'german': ['german', 'de', 'deutsch'],
      'arabic': ['arabic', 'ar', 'عربي']
    }


    self.objective_mapping = {
      'increase_sales': ['increase sales', 'sales', 'increase sales'],
      'product_launch': ['product launch', 'launch', 'product launch'],
      'brand_awareness': ['brand awareness', 'awareness', 'brand'],
      'lead_generation': ['lead generation', 'leads', 'lead gen'],
    }

  def _init_quick_lookups(self):

    self.interest_lookup = {}
    for category, keywords in self.interest_categories.items():
      for keyword in keywords:
        self.interest_lookup[keyword] = category

    self.platform_lookup = {}
    for platform, keywords in self.platform_mapping.items():
      for keyword in keywords:
        self.platform_lookup[keyword] = platform

    self.language_lookup = {}
    for language, keywords in self.language_mapping.items():
      for keyword in keywords:
        self.language_lookup[keyword] = language

    self.objective_lookup = {}
    for objective, keywords in self.objective_mapping.items():
      for keyword in keywords:
        self.objective_lookup[keyword] = objective

  @staticmethod
  def round_to_three_decimals(value: float) -> float:

    if pd.isna(value):
      return 0.0
    try:
      return round(float(value), 3)
    except:
      return 0.0

  @staticmethod
  def clean_text(text: str) -> str:

    if pd.isna(text):
      return ""
    return str(text).strip()

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

  def show_progress(self, current: int, total: int, prefix: str = ""):

    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    sys.stdout.write(f'\r{prefix} |{bar}| {current}/{total} ({percent:.1f}%)')
    sys.stdout.flush()

    if current == total:
      print()

  def generate_company_id(self, df: pd.DataFrame) -> pd.DataFrame:
    print("Generating Company IDs using categorical codes...")

    df_copy = df.copy()

    if 'Company' in df_copy.columns:
      df_copy['Company_ID_Code'] = (
              df_copy["Company"]
              .astype("category")
              .cat.set_categories(df_copy["Company"].unique(), ordered=True)
              .cat.codes + 1
      )

      df_copy['Company_ID'] = df_copy['Company_ID_Code'].apply(lambda x: f"CMP{x:06d}")

      unique_companies = df_copy[['Company', 'Company_ID']].drop_duplicates()
      for _, row in unique_companies.iterrows():
        company_name = str(row['Company'])
        company_id = row['Company_ID']
        self.company_mapping[company_name] = {
          'Company_ID': company_id,
          'Company_Name': company_name
        }

      unique_companies_count = df_copy['Company_ID'].nunique()
      print(f"✓ Generated {unique_companies_count} unique Company IDs")

      df_copy = df_copy.drop(columns=['Company_ID_Code'])
    else:
      df_copy['Company_ID'] = self.default_values['company_id']
      print("⚠ No 'Company' column found, using default Company_ID")

    return df_copy

  def generate_campaign_id(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Campaign_ID using the specified complex logic:
    1. Sort by Date
    2. Track active campaigns per Company_ID + Campaign_Goal
    3. New campaign if same platform repeats
    4. Same campaign if new platform
    5. Format as CAMP000001
    """
    print("Generating Campaign IDs using complex tracking logic...")

    df_copy = df.copy()
    required_cols = ['Date', 'Company_ID', 'Campaign_Goal', 'Channel_Used']

    for col in required_cols:
      if col not in df_copy.columns:
        print(f"⚠ Warning: '{col}' column not found, using default")
        if col == 'Company_ID':
          df_copy[col] = self.default_values['company_id']
        else:
          df_copy[col] = self.default_values.get(col.lower().replace(' ', '_'), 'unknown')

    # Step 1: Sort by Date
    print("  Step 1: Sorting by Date...")
    df_copy = df_copy.sort_values("Date").reset_index(drop=True)

    # Step 2: Initialize tracking variables
    print("  Step 2: Tracking campaigns...")
    new_ids = []
    current_id = 0

    # Store active campaigns per (Company_ID + Goal)
    # Each entry has: {"id": campaign_id, "platforms": set_of_platforms}
    active_campaigns = {}

    # Track progress
    total_rows = len(df_copy)

    # Step 3: Iterate through sorted data
    print("  Step 3: Processing rows...")
    for idx, row in df_copy.iterrows():
      company_id = str(row["Company_ID"])
      goal = str(row["Campaign_Goal"])
      platform = str(row["Channel_Used"])

      # Create unique key for this company_id and goal
      key = (company_id, goal)

      if key not in active_campaigns:
        # First time we see this Company_ID + Goal combination
        current_id += 1
        active_campaigns[key] = {
          "id": current_id,
          "platforms": {platform}
        }
        new_ids.append(current_id)
      else:
        # We've seen this Company_ID + Goal before
        if platform in active_campaigns[key]["platforms"]:
          # Same platform repeated → new campaign
          current_id += 1
          active_campaigns[key] = {
            "id": current_id,
            "platforms": {platform}  # Reset to only this platform
          }
          new_ids.append(current_id)
        else:
          # New platform → same campaign
          active_campaigns[key]["platforms"].add(platform)
          new_ids.append(active_campaigns[key]["id"])

      # Show progress
      if (idx + 1) % 100 == 0 or (idx + 1) == total_rows:
        self.show_progress(idx + 1, total_rows, "Processing")

    # Step 4: Assign the new IDs
    df_copy["Campaign_ID"] = new_ids

    # Step 5: Format as CAMP000001
    print("\n  Step 4: Formatting IDs...")
    df_copy['Campaign_ID'] = df_copy['Campaign_ID'].apply(lambda x: f"CAMP{x:06d}")

    print("  Step 5: Storing campaign mappings...")
    for idx, campaign_id in enumerate(new_ids):
      company_id = str(df_copy.iloc[idx]["Company_ID"])
      goal = str(df_copy.iloc[idx]["Campaign_Goal"])
      platform = str(df_copy.iloc[idx]["Channel_Used"])
      date = str(df_copy.iloc[idx]["Date"])

      campaign_key = f"{company_id}_{goal}_{platform}_{date}"
      self.campaign_mapping[campaign_key] = {
        'campaign_id': campaign_id,
        'company_id': company_id,
        'goal': goal,
        'platform': platform,
        'date': date
      }

    unique_campaigns_count = df_copy['Campaign_ID'].nunique()
    print(f"✓ Generated {unique_campaigns_count} unique Campaign IDs")

    # Display detailed explanation
    print("\n" + "-" * 60)
    print("CAMPAIGN ID LOGIC EXPLANATION")
    print("-" * 60)
    print("1. Data sorted chronologically by Date")
    print("2. For each (Company_ID + Campaign_Goal) combination:")
    print("   • First occurrence: Create new campaign")
    print("   • If same platform appears again: Create NEW campaign")
    print("   • If different platform appears: Add to SAME campaign")
    print("3. Format IDs as CAMP000001, CAMP000002, etc.")


    self.analyze_campaign_patterns(df_copy)

    return df_copy

  def analyze_campaign_patterns(self, df: pd.DataFrame):
    print("\n" + "-" * 60)
    print("CAMPAIGN PATTERN ANALYSIS")
    print("-" * 60)

    campaign_groups = {}
    for _, row in df.iterrows():
      campaign_id = row['Campaign_ID']
      company_id = row['Company_ID']
      goal = row['Campaign_Goal']
      platform = row['Channel_Used']
      date = row['Date']

      if campaign_id not in campaign_groups:
        campaign_groups[campaign_id] = {
          'company_id': company_id,
          'goal': goal,
          'platforms': set(),
          'dates': [],
          'row_count': 0
        }

      campaign_groups[campaign_id]['platforms'].add(platform)
      campaign_groups[campaign_id]['dates'].append(date)
      campaign_groups[campaign_id]['row_count'] += 1


    total_campaigns = len(campaign_groups)
    single_platform_campaigns = sum(1 for info in campaign_groups.values() if len(info['platforms']) == 1)
    multi_platform_campaigns = total_campaigns - single_platform_campaigns

    print(f"Total Campaigns: {total_campaigns}")
    print(f"Single-platform campaigns: {single_platform_campaigns}")
    print(f"Multi-platform campaigns: {multi_platform_campaigns}")


    sorted_campaigns = sorted(campaign_groups.items(),
                              key=lambda x: len(x[1]['platforms']),
                              reverse=True)[:5]

    print("\nTop 5 campaigns by number of platforms:")
    for campaign_id, info in sorted_campaigns:
      platforms = list(info['platforms'])
      print(f"  {campaign_id}: {len(platforms)} platforms ({', '.join(platforms)})")


    print("\nDetailed Campaign Examples:")
    example_count = 0
    for campaign_id, info in campaign_groups.items():
      if example_count >= 3:
        break
      if len(info['platforms']) > 1:
        print(f"\n  Campaign {campaign_id}:")
        print(f"    Company_ID: {info['company_id']}")
        print(f"    Goal: {info['goal']}")
        print(f"    Platforms: {len(info['platforms'])} ({', '.join(sorted(info['platforms']))})")
        print(f"    Rows in campaign: {info['row_count']}")
        print(f"    Date range: {min(info['dates'])} to {max(info['dates'])}")
        example_count += 1

  def get_duration_days(self, duration_str):

    if pd.isna(duration_str):
      return self.default_values['duration']

    try:
      match = re.search(r'(\d+)', str(duration_str))
      return int(match.group(1)) if match else self.default_values['duration']
    except:
      return self.default_values['duration']

  def parse_date(self, date_str):

    if pd.isna(date_str):
      return None

    date_str = str(date_str).strip()


    date_formats = [
      '%Y-%m-%d',  # 2022-12-31
      '%m/%d/%Y',  # 12/31/2022
      '%d/%m/%Y',  # 31/12/2022
      '%Y/%m/%d',  # 2022/12/31
      '%m-%d-%Y',  # 12-31-2022
      '%d-%m-%Y',  # 31-12-2022
      '%b %d, %Y',  # Dec 31, 2022
      '%B %d, %Y',  # December 31, 2022
      '%d %b %Y',  # 31 Dec 2022
      '%d %B %Y',  # 31 December 2022
      '%Y%m%d',  # 20221231
      '%m/%d/%y',  # 12/31/22
      '%d/%m/%y',  # 31/12/22
    ]

    for date_format in date_formats:
      try:
        return datetime.strptime(date_str, date_format)
      except ValueError:
        continue

    # Try to extract date from messy strings
    date_patterns = [
      r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # dd/mm/yyyy or mm/dd/yyyy
      r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # yyyy/mm/dd
      r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})',  # dd/mm/yy or mm/dd/yy
    ]

    for pattern in date_patterns:
      match = re.search(pattern, date_str)
      if match:
        groups = match.groups()
        if len(groups) == 3:
          # Try to determine the format
          if len(groups[0]) == 4:  # yyyy/mm/dd format
            try:
              return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
            except:
              continue
          else:
            # Try mm/dd/yyyy first (common in US data)
            try:
              return datetime(int(groups[2]), int(groups[0]), int(groups[1]))
            except:
              # Try dd/mm/yyyy
              try:
                return datetime(int(groups[2]), int(groups[1]), int(groups[0]))
              except:
                continue

    return None

  def get_age_group(self, target_audience):

    if pd.isna(target_audience):
      return self.default_values['age_group']

    audience = str(target_audience).lower()

    if '18-24' in audience or '18 to 24' in audience:
      return '18-24'
    elif '25-34' in audience or '25 to 34' in audience:
      return '25-34'
    elif '35-44' in audience or '35 to 44' in audience:
      return '35-44'
    elif '45-60' in audience or '45 to 60' in audience:
      return '45-60'
    elif 'all ages' in audience or 'all' in audience:
      return 'All Ages'

    return self.default_values['age_group']

  def get_gender(self, target_audience):
    if pd.isna(target_audience):
      return self.default_values['gender']

    audience = str(target_audience).lower()

    if 'women' in audience or 'female' in audience:
      return "Female"
    elif 'men' in audience or 'male' in audience:
      return "Male"
    elif 'all' in audience or 'all ages' in audience:
      return "All"

    return self.default_values['gender']

  def categorize_interest(self, text):

    if pd.isna(text):
      return self.default_values['interest']

    text_lower = str(text).lower().strip()

    # Check each keyword in the lookup
    for keyword, category in self.interest_lookup.items():
      if keyword in text_lower:
        return category


    return self.normalize_text(text)

  def categorize_platform(self, text):

    if pd.isna(text):
      return self.default_values['platform']

    text_lower = str(text).lower().strip()


    for keyword, platform in self.platform_lookup.items():
      if keyword in text_lower:
        return platform


    return self.normalize_text(text)

  def categorize_language(self, text):

    if pd.isna(text):
      return self.default_values['language']

    text_lower = str(text).lower().strip()

    if 'french' in text_lower or 'fr' in text_lower or 'français' in text_lower:
      return 'french'

    for keyword, language in self.language_lookup.items():
      if keyword in text_lower:
        return language

    return self.normalize_text(text)

  def categorize_objective(self, text):

    if pd.isna(text):
      return self.default_values['objective']

    text_lower = str(text).lower().strip()

    for keyword, objective in self.objective_lookup.items():
      if keyword in text_lower:
        return objective

    return self.normalize_text(text)

  def clean_acquisition_cost(self, cost_str):
    if pd.isna(cost_str):
      return 0.0

    try:
      # Remove non-numeric characters except decimal point
      cost_clean = re.sub(r'[^\d.]', '', str(cost_str))
      cost = float(cost_clean)
      return self.round_to_three_decimals(cost)
    except:
      return 0.0

  def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:

    df_copy = df.copy()

    df_copy['Acquisition_Cost'] = df_copy['Acquisition_Cost'].apply(self.clean_acquisition_cost)

    df_copy['total_budget'] = df_copy.groupby('Campaign_ID')['Acquisition_Cost'].transform('sum')

    df_copy['gender'] = df_copy['Target_Audience'].apply(
      lambda x: 'Female' if 'women' in str(x).lower() else 'Male' if 'men' in str(x).lower() else 'All'
    )

    df_copy['age_group'] = df_copy['Target_Audience'].apply(self.get_age_group)

    df_copy['interest'] = df_copy['Customer_Segment'].apply(self.categorize_interest)

    df_copy['objective'] = df_copy['Campaign_Goal'].apply(self.categorize_objective)

    df_copy['company_name'] = df_copy['Company']

    df_copy['conversions'] = df_copy['Clicks'] * df_copy['Conversion_Rate']

    df_copy['revenue'] = (df_copy['ROI'] * df_copy['Acquisition_Cost']) + df_copy['Acquisition_Cost']

    df_copy['cpc'] = df_copy.apply(
      lambda row: row['Acquisition_Cost'] / row['Clicks'] if row['Clicks'] > 0 else 0,
      axis=1
    )

    df_copy['cpa'] = df_copy.apply(
      lambda row: row['Acquisition_Cost'] / row['conversions'] if row['conversions'] > 0 else 0,
      axis=1
    )

    df_copy['ctr'] = df_copy.apply(
      lambda row: (row['Clicks'] / row['Impressions']) * 100 if row['Impressions'] > 0 else 0,
      axis=1
    )

    df_copy['budget_spent'] = df_copy['Acquisition_Cost']

    df_copy['expected_budget'] = df_copy.groupby('Campaign_ID')['Acquisition_Cost'].transform('sum') * 1.2

    numeric_cols = ['conversions', 'revenue', 'cpc', 'cpa', 'ctr', 'budget_spent', 'expected_budget']
    for col in numeric_cols:
      df_copy[col] = df_copy[col].apply(self.round_to_three_decimals)

    return df_copy


  def transform_data(self) -> pd.DataFrame:

    print("=" * 60)
    print("STARTING DATA TRANSFORMATION")
    print("=" * 60)

    total_rows = len(self.df)
    print(f"Processing {total_rows} rows...")

    print("\n[Phase 1/4] Generating Company IDs...")
    df_with_company_ids = self.generate_company_id(self.df)

    print("\n[Phase 2/4] Generating Campaign IDs...")
    df_with_campaign_ids = self.generate_campaign_id(df_with_company_ids)

    print("\n[Phase 3/4] Calculating metrics...")
    df_with_metrics = self.calculate_metrics(df_with_campaign_ids)

    print("\n[Phase 4/4] Transforming to final format...")
    transformed_rows = []

    for idx, row in df_with_metrics.iterrows():
      campaign_id = row.get('Campaign_ID', self.default_values['campaign_id'])
      company_id = row.get('Company_ID', self.default_values['company_id'])

      total_budget = self.round_to_three_decimals(row.get('total_budget', 0))
      budget_spent = self.round_to_three_decimals(row.get('budget_spent', 0))
      expected_budget = self.round_to_three_decimals(row.get('expected_budget', 0))

      conversion_rate = self.round_to_three_decimals(float(row.get('Conversion_Rate', 0)))
      roi = self.round_to_three_decimals(float(row.get('ROI', 0)))
      impressions = int(row.get('Impressions', 0))
      clicks = int(row.get('Clicks', 0))
      engagement_score = int(row.get('Engagement_Score', 0))

      conversions = self.round_to_three_decimals(row.get('conversions', 0))
      revenue = self.round_to_three_decimals(row.get('revenue', 0))
      cpc = self.round_to_three_decimals(row.get('cpc', 0))
      cpa = self.round_to_three_decimals(row.get('cpa', 0))
      ctr = self.round_to_three_decimals(row.get('ctr', 0))

      age_group = row.get('age_group', self.default_values['age_group'])
      gender = row.get('gender', self.default_values['gender'])
      interest = row.get('interest', self.default_values['interest'])

      company = self.clean_text(row.get('Company', ''))
      campaign_goal = self.clean_text(row.get('Campaign_Goal', ''))
      channel = self.clean_text(row.get('Channel_Used', ''))
      campaign_name = f"{self.normalize_text(company)}_{self.normalize_text(campaign_goal)}_{self.normalize_text(channel)}"

      platform = self.categorize_platform(row.get('Channel_Used', ''))

      start_date = self.default_values['start_date']
      end_date = self.default_values['end_date']
      parsed_date = None

      date_value = row.get('Date')
      if date_value is not None and pd.notna(date_value):
        parsed_date = self.parse_date(date_value)
        if parsed_date:
          start_date = parsed_date.strftime('%m/%d/%Y')

          duration_days = self.get_duration_days(row.get('Duration'))
          try:
            end_date_obj = parsed_date + timedelta(days=duration_days)
            end_date = end_date_obj.strftime('%m/%d/%Y')
          except:
            end_date = self.default_values['end_date']

      status = self.default_values['status']
      if end_date != self.default_values['end_date'] and parsed_date:
        try:
          end_date_obj = datetime.strptime(end_date, '%m/%d/%Y')
          today = datetime.now()
          if end_date_obj < today:
            status = "completed"
          elif end_date_obj > today:
            status = "active"
          else:
            status = "ending_today"
        except:
          status = self.default_values['status']

      language = self.categorize_language(row.get('Language'))
      location = str(row.get('Location', self.default_values['location']))
      objective = self.categorize_objective(row.get('Campaign_Goal'))
      company_name = row.get('company_name', self.default_values['company'])

      transformed_row = {
        'campaign_ID': campaign_id,
        'Campaign_Name': campaign_name,
        'Platform_Name': platform,
        'Start_Date': start_date,
        'End_Date': end_date,
        'interest': interest,
        'Total_Budget': total_budget,
        'Budget_Spent': budget_spent,
        'Expected_Budget': expected_budget,
        'Impressions': impressions,
        'Clicks': clicks,
        'Conversions': conversions,
        'Revenue': revenue,
        'CTR': ctr,
        'CPC': cpc,
        'CPA': cpa,
        'Conversion_Rate': self.round_to_three_decimals(conversion_rate * 100),  # Convert to percentage
        'engagement_score': engagement_score,
        'ROI': roi,
        'Location': location,
        'Age_Group': age_group,
        'duration_days': self.get_duration_days(row.get('Duration')),
        'Gender': gender,
        'Objective': objective,
        'Language': language,
        'Company_ID': company_id,
        'Company_Name': company_name,
        'status': status,
      }

      transformed_rows.append(transformed_row)

      if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
        self.show_progress(idx + 1, total_rows, "Transforming rows")

    transformed_df = pd.DataFrame(transformed_rows)
    self.cleaned_df = transformed_df

    print(f"\n✓ Transformation completed! Processed {total_rows} rows")
    print(f"✓ Generated {len(self.company_mapping)} unique company IDs")
    print(f"✓ Generated {len(self.campaign_mapping)} unique campaign entries")

    self.display_summary()

    return transformed_df

  def display_summary(self):
    if self.cleaned_df is None:
      print("No data to display summary")
      return

    print("\n" + "=" * 60)
    print("TRANSFORMATION SUMMARY")
    print("=" * 60)

    print(f"Total rows processed: {len(self.cleaned_df):,}")
    print(f"Unique campaigns: {self.cleaned_df['campaign_ID'].nunique():,}")
    print(f"Unique companies: {self.cleaned_df['Company_ID'].nunique():,}")

    total_budget = self.cleaned_df['Total_Budget'].sum()
    total_spent = self.cleaned_df['Budget_Spent'].sum()
    total_revenue = self.cleaned_df['Revenue'].sum()

    print(f"\nFinancial Summary:")
    print(f"  Total Budget: ${total_budget:,.2f}")
    print(f"  Total Spent: ${total_spent:,.2f}")
    print(f"  Total Revenue: ${total_revenue:,.2f}")

    status_counts = self.cleaned_df['status'].value_counts()
    print(f"\nCampaign Status:")
    for status, count in status_counts.items():
      print(f"  {status}: {count:,}")

    platform_counts = self.cleaned_df['Platform_Name'].value_counts()
    print(f"\nPlatform Distribution:")
    for platform, count in platform_counts.head(5).items():
      print(f"  {platform}: {count:,}")

    print(f"\nSample of transformed data (first 3 rows):")
    print(self.cleaned_df.head(3).to_string())

    print(f"\nFirst Campaign ID: {self.cleaned_df['campaign_ID'].iloc[0]}")
    print(f"Last Campaign ID: {self.cleaned_df['campaign_ID'].iloc[-1]}")
    print(f"First Company ID: {self.cleaned_df['Company_ID'].iloc[0]}")
    print(f"Last Company ID: {self.cleaned_df['Company_ID'].iloc[-1]}")


  def save_cleaned_data(self, output_path: str):
    if self.cleaned_df is None:
      self.transform_data()

    self.cleaned_df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned data saved to: {output_path}")
    print(f"  File contains {len(self.cleaned_df)} rows and {len(self.cleaned_df.columns)} columns")


def main():
  input_file = "D:\\mariam\\Grad Pro\\Work\\grad\\old data\\Social_Media_Advertising.csv"
  output_file = "D:\\mariam\\Grad Pro\\Work\\grad\\new data\\Social_Media_Advertising_Transformed.csv"

  try:
    import time
    start_time = time.time()

    print("=" * 60)
    print("SOCIAL MEDIA ADVERTISING DATA CLEANER")
    print("=" * 60)

    print(f"Loading data from: {input_file}")
    cleaner = SocialMediaDataCleaner(file_path=input_file)
    print(f"✓ Loaded {len(cleaner.df)} rows")

    print("\nStarting data transformation...")
    cleaned_data = cleaner.transform_data()

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\n✓ Processing completed in {processing_time:.2f} seconds")
    print(f"  ({processing_time / len(cleaner.df):.4f} seconds per row)")

    # Save cleaned data
    cleaner.save_cleaned_data(output_file)

  except FileNotFoundError:
    print(f"✗ Error: Input file '{input_file}' not found.")
  except Exception as e:
    print(f"✗ Error during cleaning: {str(e)}")
    import traceback
    traceback.print_exc()


if __name__ == "__main__":
  main()


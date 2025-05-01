import requests
import pandas as pd
import time
import os
from datetime import datetime

# Base URL for Ergast API
BASE_URL = "http://ergast.com/api/f1"

def fetch_race_results(year, round_num=None):
    """
    Fetch race results for a specific year and round (optional).
    If round is None, fetch all races for that year.
    """
    try:
        if round_num:
            url = f"{BASE_URL}/{year}/{round_num}/results.json"
        else:
            url = f"{BASE_URL}/{year}/results.json"
            
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        races = data['MRData']['RaceTable']['Races']
        
        results = []
        for race in races:
            race_name = race['raceName']
            circuit_name = race['Circuit']['circuitName']
            date = race['date']
            
            for result in race['Results']:
                driver_id = result['Driver']['driverId']
                driver_name = f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
                constructor = result['Constructor']['name']
                position = result.get('position', 'Unknown')
                
                # Handle DNF (Did Not Finish) cases
                try:
                    position = int(position)
                    status = "Finished"
                    points = float(result.get('points', 0))
                except ValueError:
                    position = None
                    status = result.get('status', 'Unknown')
                    points = 0
                
                results.append({
                    'race_name': race_name,
                    'circuit_name': circuit_name,
                    'date': date,
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'constructor': constructor,
                    'position': position,
                    'status': status,
                    'points': points,
                    'year': year,
                    'round': race['round']
                })
        
        return pd.DataFrame(results)
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching race results: {e}")
        return pd.DataFrame()

def fetch_qualifying_results(year, round_num=None):
    """
    Fetch qualifying results for a specific year and round (optional).
    If round is None, fetch all qualifying sessions for that year.
    """
    try:
        if round_num:
            url = f"{BASE_URL}/{year}/{round_num}/qualifying.json"
        else:
            url = f"{BASE_URL}/{year}/qualifying.json"
            
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        races = data['MRData']['RaceTable']['Races']
        
        results = []
        for race in races:
            race_name = race['raceName']
            circuit_name = race['Circuit']['circuitName']
            date = race['date']
            
            for qualifying in race.get('QualifyingResults', []):
                driver_id = qualifying['Driver']['driverId']
                driver_name = f"{qualifying['Driver']['givenName']} {qualifying['Driver']['familyName']}"
                constructor = qualifying['Constructor']['name']
                position = qualifying.get('position', 'Unknown')
                
                # Handle DNQ (Did Not Qualify) cases
                try:
                    position = int(position)
                    q1_time = qualifying.get('Q1', None)
                    q2_time = qualifying.get('Q2', None) 
                    q3_time = qualifying.get('Q3', None)
                except ValueError:
                    position = None
                    q1_time = None
                    q2_time = None
                    q3_time = None
                
                results.append({
                    'race_name': race_name,
                    'circuit_name': circuit_name,
                    'date': date,
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'constructor': constructor,
                    'qualifying_position': position,
                    'q1_time': q1_time,
                    'q2_time': q2_time,
                    'q3_time': q3_time,
                    'year': year,
                    'round': race['round']
                })
        
        return pd.DataFrame(results)
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching qualifying results: {e}")
        return pd.DataFrame()

def fetch_driver_standings(year, round_num=None):
    """
    Fetch driver standings for a specific year and round (optional).
    If round is None, fetch final standings for that year.
    """
    try:
        if round_num:
            url = f"{BASE_URL}/{year}/{round_num}/driverStandings.json"
        else:
            url = f"{BASE_URL}/{year}/driverStandings.json"
            
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        standings_lists = data['MRData']['StandingsTable']['StandingsLists']
        
        results = []
        for standings in standings_lists:
            round_num = standings['round']
            season = standings['season']
            
            for driver_standing in standings['DriverStandings']:
                driver_id = driver_standing['Driver']['driverId']
                driver_name = f"{driver_standing['Driver']['givenName']} {driver_standing['Driver']['familyName']}"
                position = int(driver_standing['position'])
                points = float(driver_standing['points'])
                wins = int(driver_standing['wins'])
                
                constructor = driver_standing['Constructors'][0]['name'] if driver_standing['Constructors'] else 'Unknown'
                
                results.append({
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'constructor': constructor,
                    'position': position,
                    'points': points,
                    'wins': wins,
                    'round': round_num,
                    'year': season
                })
        
        return pd.DataFrame(results)
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching driver standings: {e}")
        return pd.DataFrame()

def fetch_upcoming_race_data():
    """
    Fetch data about the upcoming race
    If no upcoming races are found in the current season, use the previous year's first race.
    """
    try:
        # Get the current year
        current_year = datetime.now().year
        using_historical_data = False
        
        # Get the current season schedule
        url = f"{BASE_URL}/{current_year}.json"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        races = data['MRData']['RaceTable']['Races']
        
        # Find the next race
        now = datetime.now()
        upcoming_races = [race for race in races if datetime.strptime(f"{race['date']} {race.get('time', '00:00:00Z')}", "%Y-%m-%d %H:%M:%SZ") > now]
        
        # If no upcoming races in the current season, try the previous year's first race
        if not upcoming_races:
            print(f"No upcoming races found for {current_year}, trying previous year...")
            current_year -= 1
            using_historical_data = True
            
            # Get the previous season schedule
            url = f"{BASE_URL}/{current_year}.json"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            races = data['MRData']['RaceTable']['Races']
            
            if not races:
                return {"error": "No races found for current or previous season"}
        
        # If using historical data, use the first race of the season
        # Otherwise use the next upcoming race
        next_race = races[0] if using_historical_data else upcoming_races[0]
        
        # Get historical results for the same circuit
        circuit_id = next_race['Circuit']['circuitId']
        
        # Fetch historical data for this circuit (last 5 years)
        historical_results = []
        for year in range(current_year-5, current_year):
            url = f"{BASE_URL}/{year}/circuits/{circuit_id}/results.json"
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                race_data = response.json()
                races = race_data['MRData']['RaceTable'].get('Races', [])
                
                if races:
                    historical_results.append({
                        'year': year,
                        'winner': races[0]['Results'][0]['Driver']['familyName'] if races[0]['Results'] else 'Unknown',
                        'winner_constructor': races[0]['Results'][0]['Constructor']['name'] if races[0]['Results'] else 'Unknown'
                    })
            except:
                # Skip if no data for that year
                pass
        
        return {
            'name': next_race['raceName'],
            'circuit': next_race['Circuit']['circuitName'],
            'circuit_id': circuit_id,
            'date': next_race['date'],
            'time': next_race.get('time', '00:00:00Z'),
            'round': next_race['round'],
            'year': current_year,
            'using_historical_data': using_historical_data,
            'historical_results': historical_results
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching upcoming race data: {e}")
        return {"error": str(e)}

def fetch_current_season_standings():
    """
    Fetch current season driver standings
    If no standings are available for the current season, use the previous year's standings.
    """
    try:
        # Get the current year
        current_year = datetime.now().year
        using_historical_data = False
        
        # Get current driver standings
        url = f"{BASE_URL}/{current_year}/driverStandings.json"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        standings_lists = data['MRData']['StandingsTable']['StandingsLists']
        
        # If no standings for current year, try previous year
        if not standings_lists:
            print(f"No standings available for {current_year}, trying previous year...")
            current_year -= 1
            using_historical_data = True
            
            # Get previous year's standings
            url = f"{BASE_URL}/{current_year}/driverStandings.json"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            standings_lists = data['MRData']['StandingsTable']['StandingsLists']
            
            if not standings_lists:
                return {"error": "No standings available for current or previous season"}
        
        drivers = []
        for driver_standing in standings_lists[0]['DriverStandings']:
            drivers.append({
                'position': int(driver_standing['position']),
                'driver_id': driver_standing['Driver']['driverId'],
                'driver_name': f"{driver_standing['Driver']['givenName']} {driver_standing['Driver']['familyName']}",
                'constructor': driver_standing['Constructors'][0]['name'] if driver_standing['Constructors'] else 'Unknown',
                'points': float(driver_standing['points']),
                'wins': int(driver_standing['wins'])
            })
        
        # Also get constructor standings for the same year
        url = f"{BASE_URL}/{current_year}/constructorStandings.json"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        standings_lists = data['MRData']['StandingsTable']['StandingsLists']
        
        constructors = []
        if standings_lists:
            for constructor_standing in standings_lists[0]['ConstructorStandings']:
                constructors.append({
                    'position': int(constructor_standing['position']),
                    'constructor': constructor_standing['Constructor']['name'],
                    'points': float(constructor_standing['points']),
                    'wins': int(constructor_standing['wins'])
                })
        
        return {
            'drivers': drivers,
            'constructors': constructors,
            'year': current_year,
            'using_historical_data': using_historical_data
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch F1 data: {e}")
()
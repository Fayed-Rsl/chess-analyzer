import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Union
from chessdotcom.aio import get_player_game_archives, get_country_details, Client
from chessdotcom.types import Resource, ChessDotComResponse
import time

class DataFetcher:
    def __init__(self, usernames: List[str], verbose: int = 1, tts: Union[str, float] = 'auto', chunk_size: int = 10, save_file: str = 'game_results.json'):
        """
        Initialize the DataFetcher.

        :param usernames: List of usernames to analyze.
        :param verbose: Verbosity level (0: no output, 1: summary, 2: detailed).
        :param tts: Time to sleep between requests ('auto' for automatic adjustment).
        :param chunk_size: Number of games to process in each chunk.
        :param save_file: File to save the results.
        """
        self.usernames = usernames
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.save_file = save_file
        self.save_lock = asyncio.Lock()
        self.tts = tts
        self.custom_tts_mapping = {
            1: 1e-3,
            2: 5e-3,
            6: 2e-2,
            10: 2e-2,
            15: 3e-2
        }        
        
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)
        self.logger = logging.getLogger(__name__)

        Client.aio = True
        Client.request_config["headers"]["User-Agent"] = (
            "My Python Application. "
            "Contact me at email@example.com"
        )

    async def get_player_country(self, username: str, tts: float = 0, **kwargs) -> ChessDotComResponse:
        """
        Fetch the country of a given player.

        :param username: Username of the player.
        :param tts: Time to sleep before making the request.
        :returns: ChessDotComResponse object containing the player's country.
        --> ChessDotComResponse
        """
        resource = Resource(
            uri=f"/player/{username}",
            tts=tts,
            top_level_attr="player",
            request_config=kwargs,
        )
        response = await Client.do_get_request(resource)
        return response

    async def get_country_name(self, country_url: str, tts: float = 0) -> str:
        """
        Fetch the country name from the Chess.com API using the custom endpoint.

        :param country_url: URL of the country.
        :param tts: Time to sleep before making the request.
        :returns: Country name.
        --> str
        """
        country_code = country_url.split('/')[-1]
        response = await get_country_details(country_code, tts=tts)
        return response.json['country']['name']

    async def fetch_opponent_profile(self, username: str, tts: float = 0) -> str:
        """
        Fetch the opponent's profile to get their country.

        :param username: Username of the opponent.
        :param tts: Time to sleep before making the request.
        :returns: Opponent's country name.
        --> str
        """
        profile = await self.get_player_country(username, tts=tts)
        country_url = profile.json['player'].get('country')
        if country_url:
            return await self.get_country_name(country_url, tts=tts)
        return "Unknown"

    async def get_opponent_info(self, game: Dict[str, Any], player_username: str) -> Dict[str, Any]:
        """
        Extract the opponent's username, result, and rating from the game data.

        :param game: Game data.
        :param player_username: Username of the player.
        :param tts: Time to sleep before making the request.
        :returns: Dictionary containing opponent information.
        --> Dict[str, Any]
        """

        if game['white']['username'].lower() == player_username.lower():
            opponent = game['black']
            result = game['white']['result']
            player_rating = game['white']['rating']
            player_color = 'white'
        else:
            opponent = game['white']
            result = game['black']['result']
            player_rating = game['black']['rating']
            player_color = 'black'

        opponent_username = opponent['username']
        opponent_rating = opponent['rating']
        opponent_country = await self.fetch_opponent_profile(opponent_username, tts=self.tts)

        info = {
            'opponent_country': opponent_country,
            'result': result,
            'opponent_rating': opponent_rating,
            'player_rating': player_rating,
            'player_color': player_color,
            'opponent_username': opponent_username,
        }
        return info

    async def save_results(self, archive_url: str, archive_results: List[Dict[str, Any]]):
        """
        Save results to the file with a lock to ensure data integrity.

        :param archive_url: URL of the archive.
        :param archive_results: Results to save.
        """
        async with self.save_lock:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r') as f:
                    saved_results = json.load(f)
            else:
                saved_results = {}

            saved_results[archive_url] = archive_results
            with open(self.save_file, 'w') as f:
                json.dump(saved_results, f)

    async def analyze_games(self, player_username: str):
        """
        Analyze the player's games and compute win rates against different countries.

        :param player_username: Username of the player to analyze.
        """
        archives_response = await get_player_game_archives(player_username)
        archives = archives_response.json['archives']
        archives = [archive.replace("https://api.chess.com/pub", "") for archive in archives]

        if self.verbose >= 1:
            self.logger.info(f"Analyzing a total of {len(archives)} archives for {player_username}")

        start_player = time.time()
        for archive_url in archives:
            start_archive = time.time()

            resource = Resource(uri=archive_url)
            games_response = await Client.do_get_request(resource)
            games = games_response.json['games']
            archive_results = []

            chunked_games = [games[i:i + self.chunk_size] for i in range(0, len(games), self.chunk_size)]
            for game_chunk in chunked_games:
                if os.path.exists(self.save_file):
                    with open(self.save_file, 'r') as f:
                        saved_results = json.load(f)
                        SAVED = True
                else:
                    SAVED = False
                    saved_results = {}

                if SAVED:
                    ALL_URL_SAVED = [g['url'] for g in sum(saved_results.values(), [])]
                    ALL_URL_GAME = [g['url'] for g in game_chunk]
                    if set(ALL_URL_GAME).issubset(set(ALL_URL_SAVED)):
                        continue

                tasks = [self.get_opponent_info(g, player_username) for g in game_chunk]
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    try:
                        opponent_info = await task
                    except Exception as e:
                        self.logger.error(f"Error processing task: {e}")
                        continue

                    opponent_country = opponent_info['opponent_country']
                    result = opponent_info['result']
                    opponent_rating = opponent_info['opponent_rating']

                    # keys inside game_chunk:
                    # (['url', 'pgn', 'time_control', 'end_time', 'rated', 'tcn', 'uuid',
                    # 'initial_setup', 'fen', 'time_class', 'rules', 'white', 'black', 'eco'])

                    game_info = {
                        'url': game_chunk[i]['url'],
                        'time_control': game_chunk[i]['time_control'],
                        'rated': game_chunk[i]['rated'],
                        'time_class': game_chunk[i]['time_class'],
                        'rules': game_chunk[i]['rules'],
                        'opponent_country': opponent_country,
                        'opponent_rating': opponent_rating,
                        'result': result,
                        'player_rating': opponent_info['player_rating'],
                        'player_color': opponent_info['player_color'],
                        'archive_url': archive_url,
                        'opponent_username': opponent_info['opponent_username'],
                        'player_username': player_username
                    }
                    archive_results.append(game_info)

            SAVING = bool(archive_results)
            if self.verbose >= 2:
                self.logger.info(f"Finished fetching games from {archive_url} in {time.time() - start_archive:.2f} seconds. Saving results: {SAVING}")
            if SAVING:
                await self.save_results(archive_url, archive_results)

        if self.verbose >= 1:
            self.logger.info(f"Finished fetching all games for {player_username} in {time.time() - start_player:.2f} seconds.")

    async def main(self):
        """
        Main function to start the analysis for all usernames.
        """
        if self.tts == 'auto':
            self.auto_tts()
        
        self.logger.info(f"Using tts={self.tts} for {len(self.usernames)} usernames")
        tasks = [self.analyze_games(username) for username in self.usernames]
        await asyncio.gather(*tasks)

    def auto_tts(self):
        """
        Adjust the tts (time to sleep) value to handle rate limiting based on the number of usernames.
        """
        num_usernames = len(self.usernames)
        # Find the closest lower bound key in custom_tts_mapping
        tts = max([key for key in self.custom_tts_mapping.keys() if key <= num_usernames], default=1)  # Default to smallest if no exact match    
        self.tts = self.custom_tts_mapping[tts]

if __name__ == "__main__":
    usernames = ["Yedfa"]  # usernames = ["Yedfa", "libetue", "Koulario", "OrBaNane", "MegaTriso", "rylessdragnir", "zino69120", "ayreonm69", "Nolzberg"] 
    analyzer = DataFetcher(usernames, verbose=2)
    asyncio.run(analyzer.main())

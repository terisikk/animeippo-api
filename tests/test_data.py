MAL_USER_LIST = {
    "data": [
        {
            "node": {
                "id": 30,
                "title": "Neon Genesis Evangelion",
                "main_picture": {
                    "medium": "https://api-cdn.myanimelist.net/images/anime/1314/108941.jpg",
                    "large": "https://api-cdn.myanimelist.net/images/anime/1314/108941l.jpg",
                },
                "nsfw": "white",
                "genres": [
                    {"id": 1, "name": "Action"},
                    {"id": 5, "name": "Avant Garde"},
                    {"id": 46, "name": "Award Winning"},
                    {"id": 8, "name": "Drama"},
                    {"id": 18, "name": "Mecha"},
                    {"id": 40, "name": "Psychological"},
                    {"id": 24, "name": "Sci-Fi"},
                    {"id": 41, "name": "Suspense"},
                ],
                "studios": [
                    {"id": 1, "name": "Test Studio 1"},
                    {"id": 5, "name": "Test Studio 2"},
                ],
                "media_type": "tv",
            },
            "list_status": {
                "status": "completed",
                "score": 10,
                "num_episodes_watched": 26,
                "is_rewatching": False,
                "updated_at": "2013-08-09T06:11:09+00:00",
                "tags": [],
            },
        },
        {
            "node": {
                "id": 270,
                "title": "Hellsing",
                "main_picture": {
                    "medium": "https://api-cdn.myanimelist.net/images/anime/10/19956.jpg",
                    "large": "https://api-cdn.myanimelist.net/images/anime/10/19956l.jpg",
                },
                "nsfw": "white",
                "genres": [
                    {"id": 1, "name": "Action"},
                    {"id": 50, "name": "Adult Cast"},
                    {"id": 58, "name": "Gore"},
                    {"id": 14, "name": "Horror"},
                    {"id": 42, "name": "Seinen"},
                    {"id": 37, "name": "Supernatural"},
                    {"id": 32, "name": "Vampire"},
                ],
                "studios": [
                    {"id": 1, "name": "Test Studio 1"},
                    {"id": 5, "name": "Test Studio 2"},
                ],
                "media_type": "tv",
            },
            "list_status": {
                "status": "completed",
                "score": 8,
                "num_episodes_watched": 13,
                "is_rewatching": False,
                "updated_at": "2017-05-26T18:32:10+00:00",
                "tags": [],
            },
        },
    ]
}

MAL_SEASONAL_LIST = {
    "data": [
        {
            "node": {
                "id": 50528,
                "title": "Golden Kamuy 4th Season",
                "main_picture": {
                    "medium": "https://api-cdn.myanimelist.net/images/anime/1855/128059.jpg",
                    "large": "https://api-cdn.myanimelist.net/images/anime/1855/128059l.jpg",
                },
                "genres": [
                    {"id": 1, "name": "Action"},
                    {"id": 50, "name": "Adult Cast"},
                    {"id": 2, "name": "Adventure"},
                    {"id": 13, "name": "Historical"},
                    {"id": 38, "name": "Military"},
                    {"id": 42, "name": "Seinen"},
                ],
                "studios": [
                    {"id": 1, "name": "Test Studio 1"},
                    {"id": 5, "name": "Test Studio 2"},
                ],
                "media_type": "tv",
                "my_list_status": {
                    "status": "not_watched",
                    "score": 0,
                    "num_episodes_watched": 0,
                    "is_rewatching": False,
                    "updated_at": "2013-08-09T06:11:09+00:00",
                    "tags": [],
                },
            }
        },
        {
            "node": {
                "id": 51535,
                "title": "Shingeki no Kyojin: The Final Season",
                "main_picture": {
                    "medium": "https://api-cdn.myanimelist.net/images/anime/1279/131078.jpg",
                    "large": "https://api-cdn.myanimelist.net/images/anime/1279/131078l.jpg",
                },
                "genres": [
                    {"id": 1, "name": "Action"},
                    {"id": 8, "name": "Drama"},
                    {"id": 58, "name": "Gore"},
                    {"id": 38, "name": "Military"},
                    {"id": 27, "name": "Shounen"},
                    {"id": 76, "name": "Survival"},
                    {"id": 41, "name": "Suspense"},
                ],
                "studios": [
                    {"id": 1, "name": "Test Studio 1"},
                    {"id": 5, "name": "Test Studio 2"},
                ],
                "media_type": "tv",
            }
        },
    ]
}

FORMATTED_MAL_USER_LIST = [
    {
        "id": 30,
        "title": "Neon Genesis Evangelion",
        "nsfw": "white",
        "genres": [
            "Action",
            "Avant Garde",
            "Award Winning",
            "Drama",
            "Mecha",
            "Psychological",
            "Sci-Fi",
            "Suspense",
        ],
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "media_type": "tv",
        "score": 10,
    },
    {
        "id": 270,
        "title": "Hellsing",
        "nsfw": "white",
        "genres": [
            "Action",
            "Adult Cast",
            "Gore",
            "Horror",
            "Seinen",
            "Supernatural",
            "Vampire",
        ],
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "media_type": "tv",
        "score": 8,
    },
]

FORMATTED_MAL_SEASONAL_LIST = [
    {
        "id": 50528,
        "title": "Golden Kamuy 4th Season",
        "genres": [
            "Action",
            "Adult Cast",
            "Adventure",
            "Historical",
            "Military",
            "Seinen",
        ],
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "media_type": "tv",
    },
    {
        "id": 51535,
        "title": "Shingeki no Kyojin: The Final Season",
        "genres": [
            "Action",
            "Drama",
            "Gore",
            "Military",
            "Shounen",
            "Survival",
            "Suspense",
        ],
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "media_type": "tv",
    },
]

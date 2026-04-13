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
                "start_season": {"year": 2020, "season": "winter"},
                "rating": "r",
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
                "title": "Hellsingfårs",
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
                "start_season": {"year": 2023, "season": "spring"},
                "rating": "r+",
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

ANI_USER_LIST = {
    "data": {
        "MediaListCollection": {
            "lists": [
                {
                    "name": "Custom List",
                    "isCustomList": True,
                    "isSplitCompletedList": False,
                    "status": "CURRENT",
                    "entries": [
                        {
                            "status": "CURRENT",
                            "score": 0,
                            "dateStarted": {"year": 2023, "month": 2, "day": 23},
                            "completedAt": {"year": None, "month": None, "day": None},
                            "media": {
                                "id": 131518,  # Same as Dr. STRONK: OLD WORLD to test deduplication
                                "title": {"romaji": "Custom List Anime"},
                                "genres": ["Action", "Adventure", "Comedy", "Sci-Fi"],
                                "tags": [],
                                "meanScore": 82,
                                "source": "MANGA",
                                "studios": {
                                    "edges": [
                                        {
                                            "node": {
                                                "name": "Test Studio",
                                                "isAnimationStudio": True,
                                            }
                                        },
                                        {
                                            "node": {
                                                "name": "Test Studio 2",
                                                "isAnimationStudio": False,
                                            }
                                        },
                                        {
                                            "node": {
                                                "name": "Test Studio 3",
                                                "isAnimationStudio": True,
                                            }
                                        },
                                    ]
                                },
                                "seasonYear": 2023,
                                "season": "SPRING",
                                "isAdult": False,
                            },
                        },
                    ],
                },
                {
                    "name": "Watching",
                    "isCustomList": False,
                    "isSplitCompletedList": False,
                    "status": "CURRENT",
                    "entries": [
                        {
                            "status": "CURRENT",
                            "score": 0,
                            "dateStarted": {"year": 2023, "month": 2, "day": 23},
                            "completedAt": {"year": None, "month": None, "day": None},
                            "media": {
                                "id": 131518,
                                "title": {"romaji": "Dr. STRONK: OLD WORLD"},
                                "genres": ["Action", "Adventure", "Comedy", "Sci-Fi"],
                                "tags": [
                                    {"id": 93, "rank": 95},
                                    {"id": 143, "rank": 89},
                                    {"id": 140, "rank": 88},
                                    {"id": 82, "rank": 85},
                                    {"id": 305, "rank": 84},
                                    {"id": 56, "rank": 81},
                                    {"id": 456, "rank": 80},
                                    {"id": 105, "rank": 71},
                                    {"id": 240, "rank": 60},
                                    {"id": 310, "rank": 60},
                                    {"id": 1277, "rank": 60},
                                    {"id": 186, "rank": 50},
                                    {"id": 909, "rank": 40},
                                    {"id": 812, "rank": 33},
                                    {"id": 191, "rank": 33},
                                    {"id": 111, "rank": 30},
                                    {"id": 23, "rank": 10},
                                ],
                                "meanScore": 82,
                                "source": "MANGA",
                                "studios": {
                                    "edges": [
                                        {
                                            "node": {
                                                "name": "Test Studio",
                                                "isAnimationStudio": True,
                                            }
                                        },
                                        {
                                            "node": {
                                                "name": "Test Studio 2",
                                                "isAnimationStudio": False,
                                            }
                                        },
                                        {
                                            "node": {
                                                "name": "Test Studio 3",
                                                "isAnimationStudio": True,
                                            }
                                        },
                                    ]
                                },
                                "seasonYear": 2023,
                                "season": "SPRING",
                                "isAdult": False,
                                "relations": {
                                    "edges": [
                                        {
                                            "relationType": "SEQUEL",
                                            "node": {"id": 790, "idMal": None},
                                        }
                                    ]
                                },
                                "recommendations": {
                                    "edges": [
                                        {
                                            "node": {
                                                "rating": 10,
                                                "mediaRecommendation": {"id": 144932},
                                            }
                                        },
                                        {
                                            "node": {
                                                "rating": 5,
                                                "mediaRecommendation": {"id": 145139},
                                            }
                                        },
                                    ]
                                },
                            },
                        },
                        {
                            "status": "COMPLETED",
                            "score": 7,
                            "dateStarted": {"year": 2022, "month": 2, "day": 22},
                            "completedAt": {"year": 2023, "month": 3, "day": 23},
                            "media": {
                                "id": 790,
                                "title": {"romaji": "Argo Roxy"},
                                "genres": ["Adventure", "Mystery", "Psychological", "Sci-Fi"],
                                "tags": [
                                    {"id": 93, "rank": 94},
                                    {"id": 391, "rank": 93},
                                    {"id": 536, "rank": 90},
                                    {"id": 217, "rank": 89},
                                    {"id": 108, "rank": 85},
                                    {"id": 109, "rank": 85},
                                    {"id": 240, "rank": 85},
                                    {"id": 175, "rank": 85},
                                    {"id": 456, "rank": 84},
                                    {"id": 517, "rank": 80},
                                    {"id": 104, "rank": 79},
                                    {"id": 327, "rank": 78},
                                    {"id": 226, "rank": 74},
                                    {"id": 365, "rank": 73},
                                    {"id": 1068, "rank": 73},
                                    {"id": 98, "rank": 72},
                                    {"id": 654, "rank": 70},
                                    {"id": 82, "rank": 68},
                                    {"id": 85, "rank": 64},
                                    {"id": 253, "rank": 60},
                                    {"id": 1310, "rank": 60},
                                    {"id": 779, "rank": 52},
                                    {"id": 40, "rank": 48},
                                    {"id": 144, "rank": 40},
                                    {"id": 102, "rank": 38},
                                    {"id": 285, "rank": 35},
                                    {"id": 157, "rank": 30},
                                    {"id": 1045, "rank": 30},
                                    {"id": 80, "rank": 12},
                                ],
                                "meanScore": 82,
                                "source": "MANGA",
                                "studios": {
                                    "edges": [
                                        {
                                            "node": {
                                                "name": "Test Studio",
                                                "isAnimationStudio": True,
                                            }
                                        },
                                        {
                                            "node": {
                                                "name": "Test Studio 2",
                                                "isAnimationStudio": False,
                                            }
                                        },
                                        {
                                            "node": {
                                                "name": "Test Studio 3",
                                                "isAnimationStudio": True,
                                            }
                                        },
                                    ]
                                },
                                "seasonYear": 2023,
                                "season": "SPRING",
                                "isAdult": False,
                                "relations": {
                                    "edges": [
                                        {
                                            "relationType": "PREQUEL",
                                            "node": {"id": 131518, "idMal": None},
                                        }
                                    ]
                                },
                                "recommendations": {
                                    "edges": [
                                        {
                                            "node": {
                                                "rating": 8,
                                                "mediaRecommendation": {"id": 144932},
                                            }
                                        },
                                    ]
                                },
                            },
                        },
                    ],
                },
            ]
        }
    }
}

ANI_MANGA_LIST = {
    "data": {
        "MediaListCollection": {
            "lists": [
                {
                    "name": "Reading",
                    "isCustomList": True,
                    "isSplitCompletedList": False,
                    "status": "CURRENT",
                    "entries": [
                        {
                            "status": "CURRENT",
                            "score": 0,
                            "completedAt": {"year": None, "month": None, "day": None},
                            "media": {
                                # Same as Dr. BONK: BONK BATTLES to test deduplication
                                "id": 324324,
                                "idMal": 12321,
                                "title": {"romaji": "Custom List Manga"},
                                "genres": ["Action", "Adventure", "Comedy", "Sci-Fi"],
                                "tags": [],
                                "meanScore": 82,
                            },
                        },
                    ],
                },
                {
                    "name": "Reading",
                    "isCustomList": False,
                    "isSplitCompletedList": False,
                    "status": "CURRENT",
                    "entries": [
                        {
                            "status": "CURRENT",
                            "score": 0,
                            "completedAt": {"year": None, "month": None, "day": None},
                            "media": {
                                "id": 324324,
                                "idMal": 12321,
                                "title": {"romaji": "Dr. BONK: BONK BATTLES"},
                                "genres": ["Action", "Adventure", "Comedy", "Sci-Fi"],
                                "tags": [
                                    {"id": 93, "rank": 95},
                                    {"id": 143, "rank": 89},
                                    {"id": 140, "rank": 88},
                                    {"id": 82, "rank": 85},
                                    {"id": 305, "rank": 84},
                                    {"id": 56, "rank": 81},
                                    {"id": 456, "rank": 80},
                                    {"id": 105, "rank": 71},
                                    {"id": 240, "rank": 60},
                                    {"id": 310, "rank": 60},
                                    {"id": 1277, "rank": 60},
                                    {"id": 186, "rank": 50},
                                    {"id": 909, "rank": 40},
                                    {"id": 812, "rank": 33},
                                    {"id": 191, "rank": 33},
                                    {"id": 111, "rank": 30},
                                    {"id": 23, "rank": 10},
                                ],
                                "meanScore": 82,
                            },
                        },
                        {
                            "status": "COMPLETED",
                            "score": 7,
                            "completedAt": {"year": 2023, "month": 3, "day": 23},
                            "media": {
                                "id": 790,
                                "idMal": 234,
                                "title": {"romaji": "Bergo Moxy"},
                                "genres": ["Adventure", "Mystery", "Psychological", "Sci-Fi"],
                                "tags": [
                                    {"id": 93, "rank": 94},
                                    {"id": 391, "rank": 93},
                                    {"id": 536, "rank": 90},
                                    {"id": 217, "rank": 89},
                                    {"id": 108, "rank": 85},
                                    {"id": 109, "rank": 85},
                                    {"id": 240, "rank": 85},
                                    {"id": 175, "rank": 85},
                                    {"id": 456, "rank": 84},
                                    {"id": 517, "rank": 80},
                                    {"id": 104, "rank": 79},
                                    {"id": 327, "rank": 78},
                                    {"id": 226, "rank": 74},
                                    {"id": 365, "rank": 73},
                                    {"id": 1068, "rank": 73},
                                    {"id": 98, "rank": 72},
                                    {"id": 654, "rank": 70},
                                    {"id": 82, "rank": 68},
                                    {"id": 85, "rank": 64},
                                    {"id": 253, "rank": 60},
                                    {"id": 1310, "rank": 60},
                                    {"id": 779, "rank": 52},
                                    {"id": 40, "rank": 48},
                                    {"id": 144, "rank": 40},
                                    {"id": 102, "rank": 38},
                                    {"id": 285, "rank": 35},
                                    {"id": 157, "rank": 30},
                                    {"id": 1045, "rank": 30},
                                    {"id": 80, "rank": 12},
                                ],
                            },
                            "mean_score": 50,
                        },
                    ],
                },
            ]
        }
    }
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
                "start_season": {"year": 2023, "season": "winter"},
                "rating": "r+",
                "num_list_users": 5,
            }
        },
        {
            "node": {
                "id": 51535,
                "title": "Shingeki no Kyojin: The Fake Season",
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
                "start_season": {"year": 2023, "season": "winter"},
                "rating": "r",
                "num_list_users": 6,
            }
        },
    ]
}

MAL_MANGA_LIST = {
    "data": [
        {
            "node": {
                "id": 1234,
                "title": "Daadaa dandaddaa",
                "main_picture": {
                    "medium": "https://api-cdn.myanimelist.net/images/anime/1855/128059.jpg",
                    "large": "https://api-cdn.myanimelist.net/images/anime/1855/128059l.jpg",
                },
                "genres": [
                    {"id": 1, "name": "Action"},
                    {"id": 2, "name": "Fantasy"},
                ],
                "studios": [
                    {"id": 1, "name": "Test Studio 1"},
                    {"id": 5, "name": "Test Studio 2"},
                ],
                "media_type": "manga",
                "rating": "r+",
                "mean": 5,
                "status": "finished",
                "source": "ORIGINAL",
            }
        },
        {
            "node": {
                "id": 532432,
                "title": "Rintintin: Beyond Johnson's Bed",
                "main_picture": {
                    "medium": "https://api-cdn.myanimelist.net/images/anime/1279/131078.jpg",
                    "large": "https://api-cdn.myanimelist.net/images/anime/1279/131078l.jpg",
                },
                "genres": [
                    {"id": 1, "name": "Fantasy"},
                    {"id": 8, "name": "Drama"},
                    {"id": 41, "name": "Suspense"},
                ],
                "studios": [
                    {"id": 1, "name": "Test Studio 1"},
                    {"id": 5, "name": "Test Studio 2"},
                ],
                "media_type": "manga",
                "rating": "r",
                "mean": 8,
                "status": "finished",
                "source": "ORIGINAL",
            }
        },
    ]
}

ANI_SEASONAL_LIST = {
    "data": {
        "Page": {
            "pageInfo": {
                "hasNextPage": False,
                "total": 50,
                "currentPage": 1,
                "lastPage": 1,
                "perPage": 50,
            },
            "media": [
                {
                    "id": 144932,
                    "title": {"romaji": "EDENS KNOCK-OFF 2nd Season"},
                    "season": "SPRING",
                    "seasonYear": 2023,
                    "relations": {
                        "edges": [
                            {"relationType": "PREQUEL", "node": {"id": 119683}},
                            {"relationType": "ADAPTATION", "node": {"id": 101860}},
                        ]
                    },
                    "genres": ["Action", "Adventure", "Drama", "Fantasy", "Supernatural"],
                    "tags": [{"id": 66, "rank": 90}, {"id": 34, "rank": 80}],
                    "coverImage": {"medium": "https://localhost/test.png"},
                    "popularity": 14090,
                    "directors": ["Haha Miyazaki"],
                },
                {
                    "id": 145139,
                    "title": {"romaji": "Kimetsu no Yaiba: Katanakaji no Sato-hen"},
                    "season": "SPRING",
                    "seasonYear": 2023,
                    "relations": {
                        "edges": [
                            {"relationType": "ADAPTATION", "node": {"id": 87216}},
                            {"relationType": "PREQUEL", "node": {"id": 142329}},
                        ]
                    },
                    "genres": ["Action", "Adventure", "Comedy", "Fantasy", "Sci-Fi"],
                    "tags": [{"id": 56, "rank": 85}],
                    "coverImage": {"medium": "https://localhost/test.png"},
                    "popularity": 131620,
                    "directors": ["Mago Senkai"],
                },
            ],
        }
    }
}

FORMATTED_MAL_USER_LIST = [
    {
        "id": 30,
        "title": "Pastel Exodus Evangelion",
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
        "features": [
            "Action",
            "Avant Garde",
            "Award Winning",
            "Drama",
            "Mecha",
            "Psychological",
            "Sci-Fi",
            "Suspense",
            "r",
        ],
        "rating": "r",
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "format": "TV",
        "score": 10,
        "user_status": "COMPLETED",
        "season_year": 2020,
        "season": "WINTER",
        "cover_image": "https://localhost/test.png",
        "source": "ORIGINAL",
        "cluster": 1,
        "directors": ["Haha Miyazaki"],
    },
    {
        "id": 270,
        "title": "Hellsingfårs",
        "genres": [
            "Action",
            "Adult Cast",
            "Gore",
            "Horror",
            "Seinen",
            "Supernatural",
            "Vampire",
        ],
        "features": [
            "Action",
            "Adult Cast",
            "Gore",
            "Horror",
            "Seinen",
            "Supernatural",
            "Vampire",
            "r+",
        ],
        "rating": "r+",
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "format": "TV",
        "score": 8,
        "user_status": "CURRENT",
        "season_year": 2023,
        "season": "SPRING",
        "cover_image": "https://localhost/test.png",
        "source": "MANGA",
        "cluster": 2,
        "directors": ["Mago Senkai"],
    },
]

FORMATTED_MAL_SEASONAL_LIST = [
    {
        "id": 50528,
        "title": "Copper Kamuy 4th Season",
        "genres": [
            "Action",
            "Adult Cast",
            "Adventure",
            "Historical",
            "Military",
            "Seinen",
        ],
        "features": [
            "Action",
            "Adult Cast",
            "Adventure",
            "Historical",
            "Military",
            "Seinen",
            "r+",
        ],
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "rating": "r+",
        "continuation_to": [],
        "format": "TV",
        "status": None,
        "season_year": 2023,
        "season": "WINTER",
        "popularity": 1,
        "cover_image": "https://localhost/test.png",
        "relations": [],
        "cluster": 1,
        "source": "ORIGINAL",
        "directors": [12345],
        "feature_info": [
            {"name": "Action", "rank": 100, "category": "Genre", "mood": "hype", "intensity": None},
        ],
    },
    {
        "id": 51535,
        "title": "Shingeki no Kyojin: The Fake Season",
        "genres": [
            "Action",
            "Drama",
            "Gore",
            "Military",
            "Shounen",
            "Survival",
            "Suspense",
        ],
        "features": [
            "Action",
            "Drama",
            "Gore",
            "Military",
            "Shounen",
            "Survival",
            "Suspense",
            "r",
        ],
        "continuation_to": [],
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "rating": "r",
        "format": "TV",
        "status": None,
        "seasonYear": 2023,
        "season": "WINTER",
        "popularity": 2,
        "cover_image": "https://localhost/test.png",
        "relations": [1],
        "cluster": 0,
        "source": "MANGA",
        "directors": [34567],
        "feature_info": [
            {"name": "Action", "rank": 100, "category": "Genre", "mood": "hype", "intensity": None},
            {
                "name": "Drama",
                "rank": 100,
                "category": "Genre",
                "mood": "emotional",
                "intensity": "heavy",
            },
            {
                "name": "Gore",
                "rank": 85,
                "category": "Theme-Other",
                "mood": "dark",
                "intensity": "heavy",
            },
        ],
    },
]

FORMATTED_ANI_SEASONAL_LIST = [
    {
        "id": 144932,
        "title": "EDENS KNOCK-OFF 2nd Season",
        "season_year": 2023,
        "season": "SPRING",
        "relations": [],
        "cover_image": "https://localhost/test.png",
        "popularity": 14090,
        "genres": ["Action", "Adventure", "Drama", "Fantasy", "Supernatural"],
        "tags": ["Shounen", "Super Power"],
        "status": "RELEASING",
        "features": [
            "Action",
            "Adventure",
            "Drama",
            "Fantasy",
            "Supernatural",
            "Shounen",
            "Super Power",
        ],
        "clustering_ranks": {"Shounen": 1, "Super Power": 2},
        "feature_info": [
            {"name": "Action", "rank": 100, "category": "Genre", "mood": "hype", "intensity": None},
            {
                "name": "Shounen",
                "rank": 90,
                "category": "Demographic",
                "mood": None,
                "intensity": None,
            },
        ],
        "directors": [12345],
    },
    {
        "id": 145139,
        "title": "Usotsu no Yaiba: Katanakaji no Sato-hen",
        "season_year": 2023,
        "season": "SPRING",
        "relations": [],
        "cover_image": "https://localhost/test.png",
        "popularity": 131620,
        "genres": ["Action", "Adventure", "Comedy", "Fantasy", "Sci-Fi"],
        "tags": ["Mythology", "Gore"],
        "clustering_ranks": {"Mythology": 1, "Gore": 2},
        "feature_info": [
            {"name": "Action", "rank": 100, "category": "Genre", "mood": "hype", "intensity": None},
            {
                "name": "Gore",
                "rank": 80,
                "category": "Theme-Other",
                "mood": "dark",
                "intensity": "heavy",
            },
        ],
        "status": "NOT_YET_RELEASED",
        "features": ["Action", "Adventure", "Comedy", "Fantasy", "Sci-Fi", "Mythology", "Gore"],
        "directors": [456],
        "continuation_to": [30],
    },
]

FORMATTED_ANI_USER_LIST = [
    {
        "id": 30,
        "title": "Pastel Exodus Evangelion",
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
        "features": [
            "Action",
            "Avant Garde",
            "Award Winning",
            "Drama",
            "Mecha",
            "Psychological",
            "Sci-Fi",
            "Suspense",
            "r",
        ],
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "format": "TV",
        "score": 10,
        "user_status": "COMPLETED",
        "season_year": 2020,
        "season": "WINTER",
        "cover_image": "https://localhost/test.png",
        "source": "ORIGINAL",
        "cluster": 1,
        "directors": ["Haha Miyazaki"],
        "clustering_ranks": {"Action": 1, "Avant Garde": 2},
        "feature_info": [
            {"name": "Action", "rank": 100, "category": "Genre", "mood": "hype", "intensity": None},
            {
                "name": "Drama",
                "rank": 100,
                "category": "Genre",
                "mood": "emotional",
                "intensity": "heavy",
            },
        ],
    },
    {
        "id": 270,
        "title": "Hellsingfårs",
        "genres": [
            "Action",
            "Adult Cast",
            "Gore",
            "Horror",
            "Seinen",
            "Supernatural",
            "Vampire",
        ],
        "features": [
            "Action",
            "Adult Cast",
            "Gore",
            "Horror",
            "Seinen",
            "Supernatural",
            "Vampire",
            "r+",
        ],
        "studios": [
            "Test Studio 1",
            "Test Studio 2",
        ],
        "format": "TV",
        "score": 8,
        "user_status": "CURRENT",
        "season_year": 2023,
        "season": "SPRING",
        "cover_image": "https://localhost/test.png",
        "source": "MANGA",
        "cluster": 2,
        "directors": ["Mago Senkai"],
        "clustering_ranks": {"Action": 1, "Adult Cast": 2},
        "feature_info": [
            {"name": "Action", "rank": 100, "category": "Genre", "mood": "hype", "intensity": None},
            {
                "name": "Horror",
                "rank": 100,
                "category": "Genre",
                "mood": "dark",
                "intensity": "heavy",
            },
            {
                "name": "Gore",
                "rank": 85,
                "category": "Theme-Other",
                "mood": "dark",
                "intensity": "heavy",
            },
        ],
    },
]


MAL_RELATED_ANIME = {
    "id": 30,
    "title": "Neon Genesis Evangelion",
    "main_picture": {
        "medium": "https://api-cdn.myanimelist.net/images/anime/1314/108941.jpg",
        "large": "https://api-cdn.myanimelist.net/images/anime/1314/108941l.jpg",
    },
    "related_anime": [
        {
            "node": {
                "id": 31,
                "title": "Neon Genesis Evangelion Season 0",
            },
            "relation_type": "prequel",
            "relation_type_formatted": "prequel",
        }
    ],
}

MIXED_USER_LIST_MAL = {
    "data": [
        {
            "node": {
                "id": 30,
            },
            "list_status": {
                "status": "completed",
                "score": 10,
            },
        },
        {
            "node": {
                "id": 270,
            },
            "list_status": {
                "status": "completed",
                "score": 8,
            },
        },
    ]
}


MIXED_USER_LIST_ANI = {
    "data": {
        "Page": {
            "pageInfo": {
                "hasNextPage": False,
                "total": 2,
                "currentPage": 1,
                "lastPage": 1,
                "perPage": 50,
            },
            "media": [
                {
                    "id": 130,
                    "idMal": 30,
                    "title": {"romaji": "Neon Genesis Evangelion"},
                    "genres": ["Action", "Adventure", "Drama", "Fantasy", "Supernatural"],
                    "tags": [
                        {"name": "Gore", "rank": 95, "isAdult": False, "category": "Theme-Other"}
                    ],
                    "format": "TV",
                    "duration": 24,
                    "episodes": 26,
                    "season": "SPRING",
                    "seasonYear": 2023,
                    "meanScore": 82,
                    "source": "ORIGINAL",
                    "studios": {"edges": [{"node": {"name": "Gainax", "isAnimationStudio": True}}]},
                    "staff": {
                        "edges": [{"role": "Director"}],
                        "nodes": [{"id": 12345}],
                    },
                    "coverImage": {"large": "https://localhost/test.png"},
                },
                {
                    "id": 1270,
                    "idMal": 270,
                    "title": {"romaji": "Hellsingfårs"},
                    "genres": ["Action", "Adventure", "Drama", "Fantasy", "Supernatural"],
                    "tags": [
                        {"name": "Gore", "rank": 80, "isAdult": False, "category": "Theme-Other"}
                    ],
                    "format": "TV",
                    "duration": 24,
                    "episodes": 13,
                    "season": "SPRING",
                    "seasonYear": 2023,
                    "meanScore": 75,
                    "source": "MANGA",
                    "studios": {"edges": [{"node": {"name": "Gonzo", "isAnimationStudio": True}}]},
                    "staff": {
                        "edges": [{"role": "Director"}],
                        "nodes": [{"id": 67890}],
                    },
                    "coverImage": {"large": "https://localhost/test.png"},
                },
            ],
        }
    }
}

MIXED_ANI_SEASONAL_LIST = {
    "data": {
        "media": [
            {
                "id": 144932,
                "title": {"romaji": "EDENS KNOCK-OFF 2nd Season"},
                "season": "SPRING",
                "seasonYear": 2023,
                "relations": {
                    "edges": [
                        {"relationType": "PREQUEL", "node": {"id": 119683, "idMal": 123}},
                        {"relationType": "ADAPTATION", "node": {"id": 101860, "idMal": 456}},
                    ]
                },
                "genres": ["Action", "Adventure", "Drama", "Fantasy", "Supernatural"],
                "tags": [{"name": "Gore", "rank": 90, "isAdult": False, "category": "Theme-Other"}],
                "coverImage": {"medium": "https://localhost/test.png"},
                "popularity": 14090,
            },
            {
                "id": 145139,
                "title": {"romaji": "Kimetsu no Yaiba: Katanakaji no Sato-hen"},
                "season": "SPRING",
                "seasonYear": 2023,
                "relations": {
                    "edges": [
                        {"relationType": "ADAPTATION", "node": {"id": 87216, "idMal": 1234}},
                        {"relationType": "PREQUEL", "node": {"id": 142329, "idMal": 2345}},
                    ]
                },
                "genres": ["Action", "Adventure", "Comedy", "Fantasy", "Sci-Fi"],
                "tags": [
                    {"name": "Swordplay", "rank": 85, "isAdult": False, "category": "Theme-Action"}
                ],
                "coverImage": {"medium": "https://localhost/test.png"},
                "popularity": 131620,
            },
        ],
    }
}

MIXED_MANGA_LIST_MAL = {
    "data": [
        {
            "node": {
                "id": 1234,
            },
            "list_status": {
                "status": "completed",
                "score": 9,
            },
        },
        {
            "node": {
                "id": 5678,
            },
            "list_status": {
                "status": "reading",
                "score": 7,
            },
        },
    ]
}

MIXED_MANGA_LIST_ANI = {
    "data": {
        "Page": {
            "pageInfo": {
                "hasNextPage": False,
                "total": 2,
                "currentPage": 1,
                "lastPage": 1,
                "perPage": 50,
            },
            "media": [
                {
                    "id": 91234,
                    "idMal": 1234,
                    "title": {"romaji": "Dr. BONK: BONK BATTLES"},
                    "genres": ["Action", "Adventure", "Comedy", "Sci-Fi"],
                    "tags": [
                        {
                            "name": "Superpowers",
                            "rank": 90,
                            "isAdult": False,
                            "category": "Theme-Fantasy",
                        },
                    ],
                    "meanScore": 82,
                },
                {
                    "id": 95678,
                    "idMal": 5678,
                    "title": {"romaji": "Bergo Moxy"},
                    "genres": ["Adventure", "Mystery"],
                    "tags": [
                        {"name": "Gore", "rank": 80, "isAdult": False, "category": "Theme-Other"},
                    ],
                    "meanScore": 75,
                },
            ],
        }
    }
}

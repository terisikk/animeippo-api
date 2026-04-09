# Mood and intensity classification — backend implementation plan

## Overview

Classify each seasonal anime with one or more moods and a single intensity level. These values are computed once during the scoring pipeline and stored as columns on the seasonal dataframe. They are consumed by the frontend funnel selector and potentially by lane diversity checks.

## Mood definitions

Each mood has a set of primary signal tags and secondary signal genres. A tag or genre may appear in at most one mood to prevent a single feature from pulling a show into multiple moods simultaneously.

### Chill — relaxing, low-stakes, comforting

Tags: Iyashikei, Cute Girls Doing Cute Things, Cute Boys Doing Cute Things, Family Life, Agriculture, Horticulture, Food, Camping, Outdoor Activities, Parenthood, Fishing.
Genres: Slice of Life.

### Hype — adrenaline, action spectacle, escalation

Tags: Martial Arts, Swordplay, Battle Royale, Super Power, Superhero, Kaiju, Guns, Fugitive, Spearplay, Proxy Battle, Archery.
Genres: Action, Mecha.

### Emotional — feelings, relationships, personal stakes

Tags: Tragedy, Coming of Age, Unrequited Love, Love Triangle, Rehabilitation, Found Family, Bullying, Suicide, Disability, Estranged Family, Arranged Marriage, Cohabitation, Matchmaking, Fake Relationship, Body Image, Marriage.
Genres: Drama, Romance.

### Dark — dread, horror, disturbing

Tags: Gore, Cosmic Horror, Body Horror, Denpa, Torture, Death Game, Cannibalism, Slavery, Drugs, Ero Guro, Survival, Noir, Pandemic, Eco-Horror, Human Experimentation, Brainwashing.
Genres: Horror, Thriller.

### Funny — comedy-first, absurdist, playful

Tags: Parody, Satire, Slapstick, Surreal Comedy, Manzai, Chibi.
Genres: Comedy.

### Cerebral — puzzles, strategy, intrigue, intellectual engagement

Tags: Conspiracy, Politics, Detective, Economics, Philosophy, Gambling, Crime, Artificial Intelligence, Software Development, Kingdom Management, Class Struggle, Espionage.
Genres: Mystery, Psychological.

### Adventurous — exploration, journey, wonder, world-building

Tags: Isekai, Reincarnation, Travel, Dungeon, Lost Civilization, Cultivation, Pirates, Space Opera, Steampunk, Wuxia, Reverse Isekai, Wilderness, Mountaineering, Ships.
Genres: Adventure, Fantasy, Sci-Fi.

### Sporty — competition, training, teamwork

Tags: all tags in category "Theme-Game-Sport" (Baseball, Basketball, Boxing, Cycling, Swimming, Tennis, Volleyball, etc.), E-Sports, Card Battle, Shogi, Mahjong, Go, Board Game.
Genres: Sports.

### Unmapped genres

The following genres do not map to a single mood and are excluded from mood scoring. Shows with these genres get their mood from tags alone:

- Supernatural — can be chill, dark, emotional, or adventurous depending on context.
- Mahou Shoujo — same issue, ranges from chill to dark.
- Ecchi — content descriptor, not a mood. Handle as a separate filter toggle in the funnel.
- Music — could be chill or emotional. Shows with Music genre get mood from their other tags/genres.
- Hentai — content descriptor, filter separately.

## Intensity definitions

Intensity is a separate axis from mood. Tags and genres can appear in both a mood definition and an intensity definition — these are independent classifications. A tag must not appear in both the heavy and light intensity lists.

### Heavy indicators (push toward all-in)

Tags: Tragedy, Conspiracy, Revenge, Suicide, Torture, Gore, Cosmic Horror, Body Horror, Death Game, Slavery, Terrorism, War, Military, Cannibalism, Drugs, Noir, Ero Guro, Philosophy, Pandemic, Crime, Bullying, Disability, Homeless, Human Experimentation, Brainwashing, Eco-Horror, Class Struggle.
Genres: Psychological, Thriller, Horror, Drama.

### Light indicators (push toward light)

Tags: Iyashikei, Cute Girls Doing Cute Things, Cute Boys Doing Cute Things, Chibi, Slapstick, Parody, Surreal Comedy, Satire, Agriculture, Camping, Food, Outdoor Activities, Horticulture, Manzai, Parenthood, Family Life, Fishing.
Genres: Slice of Life, Comedy.

### Moderate leaners (nudge toward heavy, weighted at 0.5)

Tags: Coming of Age, Politics, Espionage, Detective, Unrequited Love, Estranged Family, Kingdom Management, Gambling, Economics, Religion, Memory Manipulation, Dissociative Identities, Rehabilitation.
Genres: Mystery.

## Mood scoring algorithm

For each seasonal anime, compute a mood score per mood:

```
mood_score = sum(tag_weight for matching tags) + sum(GENRE_WEIGHT for matching genres)
```

Where `tag_weight` is the tag's Anilist rank divided by 100, with the existing category weight multiplier applied (Theme: 1.5, Setting: 1.5, Cast: 0.5, Cast-Main Cast: 1.5, Demographic: 1.5, Technical: 0.5, Sexual Content: 0.5). This reuses the same weighted values already computed for clustering.

`GENRE_WEIGHT` is a fixed contribution per matching genre (currently 1.0, consistent with clustering).

### Mood assignment

Assign all moods where `mood_score >= MOOD_THRESHOLD` (configurable, suggested starting value 1.0). Most shows will receive 1-2 moods. A show can be both "hype" and "dark" (e.g., Attack on Titan) or both "funny" and "chill" (e.g., Yuru Camp with comedic moments).

Shows where no mood reaches the threshold receive no mood labels. They appear in funnel results only when no mood filter is active. Do not assign a fallback "mixed" label — absence of mood is a meaningful signal that the show doesn't have a dominant mood.

### Mood score storage

Store the raw mood scores (not just the boolean assignment) so the funnel can rank within a mood. A show with mood_score 3.5 for "dark" should rank above one with 1.1 for "dark" when the user selects that mood.

## Intensity scoring algorithm

```
heavy_score = sum(tag_weight for matching heavy tags) + sum(1.0 for matching heavy genres)
light_score = sum(tag_weight for matching light tags) + sum(1.0 for matching light genres)
moderate_score = sum(tag_weight for matching moderate tags) * 0.5

intensity_raw = heavy_score + moderate_score - light_score
```

### Intensity bucketing

Use percentile-based bucketing across the seasonal catalog rather than fixed cutpoints:

- Bottom third of intensity_raw values → light
- Middle third → moderate
- Top third → all-in

Percentile bucketing adapts automatically to the catalog's distribution. If a season skews dark, thresholds shift so the three-way split remains useful.

### Edge case: skip intensity for mood-dominated cases

If a show's mood is exclusively "chill" (no other mood assigned), auto-assign intensity "light" without calculation — chill shows are inherently light and asking the intensity question about them in the funnel is pointless. Similarly, if a show is exclusively "dark" with a high mood score, it's almost certainly "all-in." This prevents the percentile bucketing from producing counterintuitive results for strongly typed shows.

## Implementation steps

### Step 1: Define mood and intensity tag/genre sets

Create a configuration module with the mood and intensity definitions as structured data. Validate at load time that no tag appears in more than one mood, and no tag appears in both heavy and light intensity sets. This catches configuration errors early.

### Step 2: Compute mood scores

For each seasonal anime, iterate over the mood definitions and compute the mood score from the show's tags and genres. Use the same pre-weighted tag values already computed for clustering. Store as a dictionary column or individual columns per mood (e.g., `mood_chill`, `mood_hype`, etc.).

### Step 3: Assign mood labels

Apply the threshold to produce a list column `moods` containing all qualifying mood names. Store alongside the raw scores.

### Step 4: Compute intensity

Compute the intensity_raw value from heavy, light, and moderate indicators. Apply the override rules for chill-only and dark-only shows. Percentile-bucket the remaining shows into thirds. Store as a string column `intensity` with values "light", "moderate", or "all-in".

### Step 5: Add to seasonal dataframe output

The mood and intensity columns are added to the scored seasonal dataframe alongside discovery_score, continuation strength, and other metadata. They are passed through to the frontend as part of the standard data payload.

## Validation

After implementation, spot-check the following:

- A known chill show (e.g., Yuru Camp) should be mood "chill", intensity "light".
- A known dark show (e.g., Made in Abyss) should be mood "dark" and/or "adventurous", intensity "all-in".
- A known hype show (e.g., Jujutsu Kaisen) should be mood "hype", intensity "moderate" or "all-in".
- A mixed-tone show (e.g., Gintama) should get multiple moods ("funny", "hype") and intensity "moderate".
- A show with sparse tags should either get no mood or a single mood from its genre alone, not multiple moods from thin evidence.
# Error analysis — 15 worst queries by ndcg@1000

Stage: `reranked`  |  run file used: `False`

## Failure-mode distribution

- **recall_miss_top100** (15): gold document not in the top-100 (R@100 = 0); without the run file we cannot tell whether it sits in 100..1000 or beyond.

## Query-side tags

- `few_entities`: 5
- `sparse_query`: 4

## Case studies

### 1. Query `1038` — recall_miss_top100 [few_entities]

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 4, proper-nouns = 0
- Gold: Smokey and the Bandit II is a 1980...
- Query: Ive had a movie on my mind for a few days I watcher several years back (at least 10+) I dont remember alot of it, but there was a guy chased by cops for superlong and at one point it was in a desert environment He mustve had several dozen of cops right behind him at one point and when it looked like he was about to get caught a trucker showed up in a far distans, shortly after several others showed up behind it and they spread out in a line Im not sure but I think he was driving an american musc...

### 2. Query `279` — recall_miss_top100 [few_entities]

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 5, proper-nouns = 1
- Gold: Freedom Writers is a 2007 American drama film...
- Query: There was a flashback scene of two young black kids They were sitting on a bench, Probably in a park or something The protagonist’s friend shows him a gun he found The gun then accidentally triggers, killing his friend The protagonist then sits beside his dead friend, until the cops arrive and take him to jail or something

### 3. Query `890` — recall_miss_top100

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 8, proper-nouns = 2
- Gold: Sanctuary is a 1961 drama film directed by...
- Query: So I saw this movie probably around 1979 It was a B & W movie set in the 1920’s Gangster era as can be told by the clothing and the cars It starts out with this young high school couple at a party and the male is getting very drunk, they both leave the party and he’s driving (no shocker there, they can’t crash unless he drives) You guessed it, they crash coming around a corner into a tree I can’t remember if they were found, but the girl ends up at this old farmhouse and being held hostage by a ...

### 4. Query `403` — recall_miss_top100

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 18, proper-nouns = 16
- Gold: V for Vendetta is a 2005 dystopian political...
- Query: -Date: I’m not entirely sure, but I think I watched it on Netflix a year or so ago -Language: I believe it was in either German, French, or English -Where seen: I think on Netflix It was definitely a movie -Extras: in color, recently made, setting somewhere in Europe (Germany, France, UK) -Description: A woman wakes up in some secret underground apartment (I can’t remember if she was kidnapped or otherwise ended up there another way) The resident of the apartment is a man who wears a mask and gl...

### 5. Query `813` — recall_miss_top100

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 15, proper-nouns = 4
- Gold: Thriller (also known as Boris Karloff's Thriller and...
- Query: Looking for movie I saw when I was a kid It may have been an episode of a TV show Anyway, this newly married couple bought an old lighthouse or at least a house that was on a cliff overlooking the sea The woman found a bunch of mirrors in the attic of the house that mesmerized her She would dance in front of them The story was a former female occupant had died in the house under suspicious circumstances, maybe she committed suicide or something But the mirrors were haunted and the ghost arose an...

### 6. Query `714` — recall_miss_top100

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 15, proper-nouns = 3
- Gold: Sorority Boys is a 2002 American comedy film...
- Query: I dont remember most of this scene in this movie I watch this movie about 2004-2005 the movie is about 3 men whose lost their money and they were suspecting a group of women So, in order to investigate the women, they disguise themselve as woman and make excuse to live with the women The scene that remember the most is where a man taking bath at night and suddenly a woman joined him But the girl didnt realizes that the man is a man caused she didnt wear her glasses And in the movie, the men fall...

### 7. Query `992` — recall_miss_top100 [sparse_query|few_entities]

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 1, proper-nouns = 0
- Gold: Eve of Destruction is a 1991 American science...
- Query: so i love movies and have found some through memories this one is a very vague one lol so a blond woman is in a hotel room with a guy i believe she is an alien and the guy is trying to get laid and she isnt interested and he takes her necklace and slides it down his pants trying to be all into it so she crawls across the bed unzips and the guy is all like yeah and she takes it in her mouth starts sliding the necklace out and bites it off then leaves lol i believe its an 90’s movie but i cant rem...

### 8. Query `832` — recall_miss_top100 [sparse_query|few_entities]

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 3, proper-nouns = 0
- Gold: Mikey is a 1992 American psychological slasher film...
- Query: All I remember of this film is that there is a kid in a house holding a bow and arrow and facing his father Father tells him to put the bow down but the kid shoots him in the belly with the arrow and kills him Any ideas?

### 9. Query `861` — recall_miss_top100

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 14, proper-nouns = 2
- Gold: Nightmares &amp; Dreamscapes: From the Stories of Stephen...
- Query: I saw this movie on TV, it was probably somewhere between 2004-2008 It might have been made in the late 90s, but I’m not sure The film was about this couple who were going somewhere for a honeymoon (I think), and traveling around this weird abandoned town Throughout the film, they see very disturbing events, the most prominent being this scarred-up white cat It keeps following them Throughout the film, the husband goes insane He’s bloodied-up and yelling all the time At the end of the film, this...

### 10. Query `345` — recall_miss_top100

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 10, proper-nouns = 2
- Gold: A Clockwork Orange is a 1971 dystopian crime...
- Query: I hope someone here can help me! it’s a movie that i saw around 6 ago on tv (not sure) its a movie about a group people (5) that Break Into a house ans accidentally kill a woman, they had weird white masks in the end of the movie one of the mans is in hospital bed I remember that the movie had a bad ending The hole movie wasn’t really happy The jumping between scenes was also very strange Rooms with to mush color and conversations about thing that didn’t fit in the movie I saw the movie when I w...

### 11. Query `902` — recall_miss_top100 [sparse_query]

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 1, proper-nouns = 4
- Gold: The Bed Sitting Room is a 1969 British...
- Query: All I remember about the movie was that what seemed like the last people alive on Earth were living in a junkyard fighting over who got the broken TV and other things – it seemed like the cast were British or Australian

### 12. Query `837` — recall_miss_top100

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 4, proper-nouns = 2
- Gold: Until the End of the World (; )...
- Query: Very late one night I saw part of a very long post-apocalyptic film (as in multiple hours), maybe filmed in Australia or about Australia I though that it ended in a cave Seen in the 80’s, I think Thanks for the help

### 13. Query `625` — recall_miss_top100 [sparse_query]

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 2, proper-nouns = 3
- Gold: Newhart is an American sitcom television series that...
- Query: Hi , Had watched a few episodes of an american comedy series probably late 80’s about a couple who own a hotel/ski lodge I remember one episode where a lady goes on a date and her date brings his tow truck and she says ” no way am i sitting in a car that i have to climb up into… (or something like that) Thank you, Sean

### 14. Query `652` — recall_miss_top100

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 4, proper-nouns = 4
- Gold: Scary Movie is a 2000 American slasher parody...
- Query: At some point, I saw a movie that ended the same way as The Usual Suspects…I’m thinking it was probably satire It was in English and in Color I’m not sure when or where I saw it, but it had to be after 1995 It could have been from a tv show as well – not 100% sure

### 15. Query `934` — recall_miss_top100 [few_entities]

- NDCG@1000 = 0.0, R@100 = 0.0, sentences = 5, proper-nouns = 1
- Gold: Fistful of Flies is a 1997 Australian film...
- Query: I vaguely remember a bit weird Australian movie about a teenage girl who argued with her parents, her mom in particular She kinda lived in her own world daydreaming and masturbating often And there was a scene where her mother told her to set the table for lunch and as she was doing it she starts to rub her vagina on the corner of the table and closes her eyes in ecstasy and then her mom catches her doing it and is outraged Think the movie is from the 90’s Also, but not sure if I remember it cor...


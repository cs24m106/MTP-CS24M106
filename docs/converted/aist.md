[ ]
[ ]

[Skip to content](#data-formats)

[![logo](../images/white.png)](.. "AIST Dance Video Database (AIST Dance DB)")

AIST Dance Video Database (AIST Dance DB)

Data Formats

Initializing search

[![logo](../images/white.png)](.. "AIST Dance Video Database (AIST Dance DB)")
AIST Dance Video Database (AIST Dance DB)

* [Home](..)
* [Terms of Use](../terms_of_use/)
* [Getting the Database](../getting_the_database/)
* [x]

  Database

  Database
  + [Database Structure](../database_structure/)
  + [ ]

    Data Formats

    [Data Formats](./)

    Table of contents
    - [Data Formats](#data-formats_1)
    - [Naming Rules](#naming-rules)
    - [Dance Genres](#dance-genres)
    - [Filming Situations](#filming-situations)
    - [Camera Positions](#camera-positions)
    - [Dancers](#dancers)
    - [Musical Pieces](#musical-pieces)
    - [Choreography](#choreography)
  + [Database Download](../database_download/)
  + [ ]

    All Links

    All Links
    - [ ]

      Refined Video

      Refined Video
      * [Basic](../all_links/all_links_refined_basic/)
      * [Advanced](../all_links/all_links_refined_advanced/)
      * [Group](../all_links/all_links_refined_group/)
      * [Moving Camera](../all_links/all_links_refined_moving/)
      * [Showcase](../all_links/all_links_refined_showcase/)
      * [Cypher](../all_links/all_links_refined_cypher/)
      * [Battle](../all_links/all_links_refined_battle/)
    - [ ]

      Raw Video

      Raw Video
      * [Basic](../all_links/all_links_raw_basic/)
      * [Advanced](../all_links/all_links_raw_advanced/)
      * [Group](../all_links/all_links_raw_group/)
      * [Moving Camera](../all_links/all_links_raw_moving/)
      * [Showcase](../all_links/all_links_raw_showcase/)
      * [Cypher](../all_links/all_links_raw_cypher/)
      * [Battle](../all_links/all_links_raw_battle/)
    - [Musical Pieces](../all_links/all_links_musical_pieces/)
  + [Dancer Info](../dancer_info/)
* [ ]

  Research

  Research
  + [How to refine raw videos](../how_to_refine_raw_videos/)
  + [List of Related Publications](../list_of_related_publications/)
  + [Tasks](../tasks/)
* [FAQ](../faq/)
* [Contact Us / Members](../contact/)

Table of contents

* [Data Formats](#data-formats_1)
* [Naming Rules](#naming-rules)
* [Dance Genres](#dance-genres)
* [Filming Situations](#filming-situations)
* [Camera Positions](#camera-positions)
* [Dancers](#dancers)
* [Musical Pieces](#musical-pieces)
* [Choreography](#choreography)

# ![](../images/black_logo.png) Data Formats[¶](#data-formats "Permanent link")

## Data Formats[¶](#data-formats_1 "Permanent link")

Videos: compressed MPEG-4 file (mp4) at 10Mbps and 2Mbps

Musical pieces: uncompressed audio file (wav) and compressed MPEG-1 Audio Layer III file (mp3)

## Naming Rules[¶](#naming-rules "Permanent link")

The naming rules of the files slightly differ depending on the filming situation of each file as follows.

File names for filming situation sSH, sCY, and sBT consist of symbols to specify the filming situation, camera position, and variation which indicate the participating dancers and the music pieces.

`eg) gBR_sBA_c01_d03_mBR3_ch04.mp4`
![symbol_1](../images/symbols/symbol_1.png)

In addition to the format above, file names for filming situation sGR concatenate the identifiers of three dancers as d01\_d02\_d03.

`eg) gBR_sBA_c01_d01_d02_d03_mBR3_ch04.mp4`
![symbol_2](../images/symbols/symbol_2.png)

File names for filming situation sSH, sCY, and sBT consist of symbols to specify the filming situation, camera position, participating dancers, and variation of the music piece.

`eg) sSH_c01_v0.mp4`
![symbol_3](../images/symbols/symbol_3.png)

The dancers who participate in each variation of a musical piece are the following.

| Dancer ID |  |  |  |  |
| --- | --- | --- | --- | --- |
| d20 | d31 | d21 | d32 | d33 |
| d34 | d35 | d22 | d23 | d24 |

| Showcase | Camera ID | Variation | Music ID | Dancer |
| --- | --- | --- | --- | --- |
| sSH | c01 | v0 | mMH0 | All the above |
| sSH | c01 | v1 | mLH2 | All the above |
| sSH | c01 | v2 | mBR4 | All the above |

| Showcase | Camera ID | Variation | Music ID | Dancer |
| --- | --- | --- | --- | --- |
| sCY | c01 | v3 | mPO0 | All the above |
|  |  |  | mMH1 |  |
|  |  |  | mBR2 |  |
|  |  |  | mPO2 |  |
|  |  |  | mPO3 |  |
|  |  |  | mLO2 |  |
|  |  |  | mLO4 |  |
|  |  |  | mPO4 |  |
|  |  |  | mBR5 |  |
|  |  |  | mLO5 |  |
|  |  |  | mPO5 |  |
|  |  |  | mHO5 |  |
|  |  |  | mHO3 |  |
| - | - | - | - | - |
| sCY | c01 | v4 | mBR5 | All the above |
|  |  |  | mHO5 |  |
|  |  |  | mLO5 |  |
|  |  |  | mHO3 |  |
|  |  |  | mBR2 |  |
|  |  |  | mMH2 |  |
|  |  |  | mPO2 |  |
|  |  |  | mLO4 |  |
|  |  |  | mPO4 |  |
|  |  |  | mBR4 |  |
|  |  |  | mPO3 |  |
|  |  |  | mLO2 |  |
|  |  |  | mBR3 |  |

| Battle | Camera ID | Dancer ID | Dancer ID | Variation | Music ID |
| --- | --- | --- | --- | --- | --- |
| sBT | c01 | d21 | d31 | v5 | mPO0 |
|  |  |  |  |  | mLH5 |
| - | - | - | - | - | - |
| sBT | c01 | d20 | d33 | v6 | mMH1 |
|  |  |  |  |  | mHO3 |
| - | - | - | - | - | - |
| sBT | c01 | d34 | d35 | v7 | mLO4 |
|  |  |  |  |  | mPO5 |

A complete list of the file names is available at [Database Download](../database_download/).

## Dance Genres[¶](#dance-genres "Permanent link")

| Genres | Symbols |
| --- | --- |
| Break | gBR |
| Pop | gPO |
| Lock | gLO |
| Middle Hip-hop | gMH |
| LA style Hip-hop | gLH |
| House | gHO |
| Waack | gWA |
| Krump | gKR |
| Street Jazz | gJS |
| Ballet Jazz | gJB |

## Filming Situations[¶](#filming-situations "Permanent link")

| Situations | Symbols |
| --- | --- |
| Basic Dance | sBM |
| Advanced Dance | sFM |
| Moving Camera | sMM |
| Group Dance | sGR |
| Showcase | sSH |
| Cypher | sCY |
| Battle | sBT |

## Camera Positions[¶](#camera-positions "Permanent link")

Fixed Camera: c01–c09

![basic_dance](../images/basic_dance.png)

Moving Camera: c10

![moving_camera](../images/moving_camera.png)

## Dancers[¶](#dancers "Permanent link")

See [Dancer Info](../dancer_info/) for detailed information on dancers d01–d35.

## Musical Pieces[¶](#musical-pieces "Permanent link")

| Genre | ID | tempo |
| --- | --- | --- |
| Break | mBR0 | 80 |
|  | mBR1 | 90 |
|  | mBR2 | 100 |
|  | mBR3 | 110 |
|  | mBR4 | 120 |
|  | mBR5 | 130 |
| - | - | - |
| Pop | mPO0 | 80 |
|  | mPO1 | 90 |
|  | mPO2 | 100 |
|  | mPO3 | 110 |
|  | mPO4 | 120 |
|  | mPO5 | 130 |
| - | - | - |
| Lock | mLO0 | 80 |
|  | mLO1 | 90 |
|  | mLO2 | 100 |
|  | mLO3 | 110 |
|  | mLO4 | 120 |
|  | mLO5 | 130 |
| - | - | - |
| Middle Hip-hop | mMH0 | 80 |
|  | mMH1 | 90 |
|  | mMH2 | 100 |
|  | mMH3 | 110 |
|  | mMH4 | 120 |
|  | mMH5 | 130 |
| - | - | - |
| LA style Hip-hop | mLH0 | 80 |
|  | mLH1 | 90 |
|  | mLH2 | 100 |
|  | mLH3 | 110 |
|  | mLH4 | 120 |
|  | mLH5 | 130 |
| - | - | - |
| House | mHO0 | 110 |
|  | mHO1 | 115 |
|  | mHO2 | 120 |
|  | mHO3 | 125 |
|  | mHO4 | 130 |
|  | mHO5 | 135 |
| - | - | - |
| Waack | mWA0 | 80 |
|  | mWA1 | 90 |
|  | mWA2 | 100 |
|  | mWA3 | 110 |
|  | mWA4 | 120 |
|  | mWA5 | 130 |
| - | - | - |
| Krump | mKR0 | 80 |
|  | mKR1 | 90 |
|  | mKR2 | 100 |
|  | mKR3 | 110 |
|  | mKR4 | 120 |
|  | mKR5 | 130 |
| - | - | - |
| Street Jazz | mJS0 | 80 |
|  | mJS1 | 90 |
|  | mJS2 | 100 |
|  | mJS3 | 110 |
|  | mJS4 | 120 |
|  | mJS5 | 130 |
| - | - | - |
| Ballet Jazz | mJB0 | 80 |
|  | mJB1 | 90 |
|  | mJB2 | 100 |
|  | mJB3 | 110 |
|  | mJB4 | 120 |
|  | mJB5 | 130 |

## Choreography[¶](#choreography "Permanent link")

A complete list of choreography titles ch01–ch10 is at  [here](../data/choreo.xlsx)  .

Copyright © 2019 National Institute of Advanced Industrial Science and Technology (AIST)

Made with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

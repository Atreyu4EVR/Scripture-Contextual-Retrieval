"""Synthetic Stack Exchange dump for tests. No real posts, users, or text.

Mirrors the real file's quirks: UTF-8 BOM, CRLF line endings, pipe-delimited
tags on most questions (the 2024-04 dump's encoding) with one angle-bracket
row (every other dump's encoding), a deleted-owner question, a pre-4.0 question
without ContentLicense, a fabricated watermark row, and a deleted post.
"""

from __future__ import annotations

import io
from pathlib import Path

import py7zr

BOM = "﻿"

USERS_XML = """<?xml version="1.0" encoding="utf-8"?>
<users>
  <row Id="-1" Reputation="1" CreationDate="2011-08-23T00:00:00.000" DisplayName="Community" />
  <row Id="10" Reputation="101" CreationDate="2012-01-01T00:00:00.000" DisplayName="Pilgrim"
       Location="Somewhere, UT" AboutMe="&lt;p&gt;never read by the parser&lt;/p&gt;" />
  <row Id="11" Reputation="55" CreationDate="2016-03-03T00:00:00.000" DisplayName="Seeker" />
</users>
"""

POSTS_XML = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" CreationDate="2019-04-02T08:30:00.000" Score="7" ViewCount="900"
       Body="&lt;p&gt;I read that there are &lt;em&gt;several&lt;/em&gt; accounts of the First
Vision.&lt;/p&gt;&lt;p&gt;How are the differences explained?&lt;/p&gt;"
       OwnerUserId="10" Title="Why are there several accounts of the First Vision?"
       Tags="|lds|joseph-smith|" AnswerCount="2" ContentLicense="CC BY-SA 4.0" />
  <row Id="2" PostTypeId="1" CreationDate="2024-02-10T12:00:00.000" Score="3" ViewCount="120"
       Body="&lt;p&gt;What does the Book of Mormon teach about baptism for the dead?&lt;/p&gt;"
       OwnerUserId="11" Title="Baptism for the dead in the Book of Mormon"
       Tags="|book-of-mormon|baptism|" AnswerCount="1" ContentLicense="CC BY-SA 4.0" />
  <row Id="3" PostTypeId="1" CreationDate="2013-06-15T09:00:00.000" Score="12" ViewCount="4000"
       Body="&lt;p&gt;How do Latter-day Saints understand the Trinity?&lt;/p&gt;"
       OwnerUserId="10" Title="Latter-day Saint view of the Trinity"
       Tags="&lt;lds&gt;&lt;trinity&gt;" AnswerCount="4" />
  <row Id="4" PostTypeId="1" CreationDate="2020-09-09T09:00:00.000" Score="1" ViewCount="50"
       Body="&lt;p&gt;Why did plural marriage end in 1890?&lt;/p&gt;"
       OwnerDisplayName="user4242" Title="End of plural marriage"
       Tags="|lds|history|" AnswerCount="0" ContentLicense="CC BY-SA 4.0" />
  <row Id="5" PostTypeId="1" CreationDate="2020-01-01T00:00:00.000" Score="2" ViewCount="70"
       Body="&lt;p&gt;What is the Catholic view of purgatory?&lt;/p&gt;"
       OwnerUserId="11" Title="Purgatory" Tags="|catholicism|purgatory|"
       AnswerCount="1" ContentLicense="CC BY-SA 4.0" />
  <row Id="6" PostTypeId="2" ParentId="1" CreationDate="2019-04-03T08:30:00.000" Score="5"
       Body="&lt;p&gt;An answer body.&lt;/p&gt;" OwnerUserId="11" ContentLicense="CC BY-SA 4.0" />
  <row Id="7" PostTypeId="1" CreationDate="2020-05-05T00:00:00.000" Score="0"
       DeletionDate="2020-06-06T00:00:00.000" OwnerUserId="10" Title="Deleted question"
       Tags="|lds|" />
  <row Id="1000000001" PostTypeId="1" CreationDate="2025-06-01T01:00:00.100" Score="0"
       ViewCount="0" Body="&lt;p&gt;watermark&lt;/p&gt;" OwnerUserId="-1" Title="Watermark"
       Tags="|lds|" AnswerCount="0" ContentLicense="CC BY-SA 4.0" />
</posts>
"""

TAGS_XML = """<?xml version="1.0" encoding="utf-8"?>
<tags>
  <row Id="94" TagName="lds" Count="4" />
  <row Id="2" TagName="catholicism" Count="1" />
</tags>
"""


def as_dump_bytes(xml: str) -> bytes:
    """BOM plus CRLF, exactly as the Internet Archive file is encoded."""
    return (BOM + xml.replace("\n", "\r\n")).encode("utf-8")


def build_dump(dest: Path) -> Path:
    """Write a .7z with the three tables the parser expects; return its path."""
    with py7zr.SevenZipFile(dest, mode="w") as archive:
        archive.writef(io.BytesIO(as_dump_bytes(USERS_XML)), "Users.xml")
        archive.writef(io.BytesIO(as_dump_bytes(POSTS_XML)), "Posts.xml")
        archive.writef(io.BytesIO(as_dump_bytes(TAGS_XML)), "Tags.xml")
    return dest

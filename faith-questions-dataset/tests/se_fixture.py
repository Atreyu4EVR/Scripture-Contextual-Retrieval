"""Synthetic Stack Exchange dump for tests. No real posts, users, or text."""

from __future__ import annotations

import io
from pathlib import Path

import py7zr

USERS_XML = """<?xml version="1.0" encoding="utf-8"?>
<users>
  <row Id="10" Reputation="101" CreationDate="2012-01-01T00:00:00.000" DisplayName="Pilgrim" />
  <row Id="11" Reputation="55" CreationDate="2016-03-03T00:00:00.000" DisplayName="Seeker" />
</users>
"""

# Five posts: three LDS-tagged questions in mixed conventions, one non-LDS
# question, one answer. Question 4 has a deleted owner; question 3 predates the
# CC BY-SA 4.0 cutover and lacks ContentLicense so the date fallback applies.
POSTS_XML = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" CreationDate="2019-04-02T08:30:00.000" Score="7" ViewCount="900"
       Body="&lt;p&gt;I read that there are &lt;em&gt;several&lt;/em&gt; accounts of the First
Vision.&lt;/p&gt;&lt;p&gt;How are the differences explained?&lt;/p&gt;"
       OwnerUserId="10" Title="Why are there several accounts of the First Vision?"
       Tags="&lt;mormonism&gt;&lt;joseph-smith&gt;" AnswerCount="2" ContentLicense="CC BY-SA 4.0" />
  <row Id="2" PostTypeId="1" CreationDate="2024-02-10T12:00:00.000" Score="3" ViewCount="120"
       Body="&lt;p&gt;What does the Book of Mormon teach about baptism for the dead?&lt;/p&gt;"
       OwnerUserId="11" Title="Baptism for the dead in the Book of Mormon"
       Tags="|book-of-mormon|baptism|" AnswerCount="1" ContentLicense="CC BY-SA 4.0" />
  <row Id="3" PostTypeId="1" CreationDate="2013-06-15T09:00:00.000" Score="12" ViewCount="4000"
       Body="&lt;p&gt;How do Latter-day Saints understand the Trinity?&lt;/p&gt;"
       OwnerUserId="10" Title="Latter-day Saint view of the Trinity"
       Tags="&lt;mormonism&gt;&lt;trinity&gt;" AnswerCount="4" />
  <row Id="4" PostTypeId="1" CreationDate="2020-09-09T09:00:00.000" Score="1" ViewCount="50"
       Body="&lt;p&gt;Why did plural marriage end in 1890?&lt;/p&gt;"
       OwnerDisplayName="user4242" Title="End of plural marriage"
       Tags="&lt;lds&gt;&lt;history&gt;" AnswerCount="0" ContentLicense="CC BY-SA 4.0" />
  <row Id="5" PostTypeId="1" CreationDate="2020-01-01T00:00:00.000" Score="2" ViewCount="70"
       Body="&lt;p&gt;What is the Catholic view of purgatory?&lt;/p&gt;"
       OwnerUserId="11" Title="Purgatory" Tags="&lt;catholicism&gt;&lt;purgatory&gt;"
       AnswerCount="1" ContentLicense="CC BY-SA 4.0" />
  <row Id="6" PostTypeId="2" ParentId="1" CreationDate="2019-04-03T08:30:00.000" Score="5"
       Body="&lt;p&gt;An answer body.&lt;/p&gt;" OwnerUserId="11" ContentLicense="CC BY-SA 4.0" />
</posts>
"""

TAGS_XML = """<?xml version="1.0" encoding="utf-8"?>
<tags>
  <row Id="1" TagName="mormonism" Count="3" />
  <row Id="2" TagName="catholicism" Count="1" />
</tags>
"""


def build_dump(dest: Path) -> Path:
    """Write a .7z with the three tables the parser expects; return its path."""
    with py7zr.SevenZipFile(dest, mode="w") as archive:
        archive.writef(io.BytesIO(USERS_XML.encode("utf-8")), "Users.xml")
        archive.writef(io.BytesIO(POSTS_XML.encode("utf-8")), "Posts.xml")
        archive.writef(io.BytesIO(TAGS_XML.encode("utf-8")), "Tags.xml")
    return dest

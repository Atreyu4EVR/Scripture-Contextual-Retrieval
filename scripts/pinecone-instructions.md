# Pinecone Instructions

Use the following Pinecone index: `standard-works` using the API key `PINECONE_API_KEY` that's stored in `.env`.

## Verse Mapping

When we upsert to the index, we must map the verse `text` to the data field `pageContent`

## Metadata Requirements

Metadata is stored in object `metadata`, with the following values:

- `context` (only contains context - NOT verse)
- `text` (the verse content - named `text` for LangChain compatibility)
- `reference`
- `volume`

## Example Recorda

Current - Bad Example:

```txt
ID: 00020eca-b541-4e3f-b315-c05dd2221ab7
values: [-0.00213053753, -0.021202486...]
metadata:
    context: "**Context**: \"The following passage is from **Exodus 26:1** from the **Old Testament**. This chapter contains 37 verses and records instructions for constructing the tabernacle: ten linen curtains with cherubim joined by loops and gold clasps; eleven goats’ hair curtains joined with brass clasps; coverings of dyed rams’ skins and badgers’ skins. It specifies shittim-wood boards, silver sockets, bars overlaid with gold, and assembly pattern. It commands making the veil and door hanging, placing the ark and mercy seat, and arranging the table and candlestick.\"\n**Verse**: \"Moreover thou shalt make the tabernacle with ten curtains of fine twined linen, and blue, and purple, and scarlet: with cherubims of cunning work shalt thou make them.\""
    reference: "Exodus 26:1"
    verse: "Moreover thou shalt make the tabernacle with ten curtains of fine twined linen, and blue, and purple, and scarlet: with cherubims of cunning work shalt thou make them."
    volume: "Old Testament"
```

Corrected - Good Example:

```txt
ID: 00020eca-b541-4e3f-b315-c05dd2221ab7
values: [-0.00213053753, -0.021202486...]
metadata:
    context: "The following passage is from **Exodus 26:1** from the **Old Testament**. This chapter contains 37 verses and records instructions for constructing the tabernacle: ten linen curtains with cherubim joined by loops and gold clasps; eleven goats' hair curtains joined with brass clasps; coverings of dyed rams' skins and badgers' skins. It specifies shittim-wood boards, silver sockets, bars overlaid with gold, and assembly pattern. It commands making the veil and door hanging, placing the ark and mercy seat, and arranging the table and candlestick."
    text: "Moreover thou shalt make the tabernacle with ten curtains of fine twined linen, and blue, and purple, and scarlet: with cherubims of cunning work shalt thou make them."
    reference: "Exodus 26:1"
    volume: "Old Testament"
```

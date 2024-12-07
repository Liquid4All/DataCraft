# 1. Language (Plain Text, e.g., Captions)

```json
{
  "text": "Your caption or text here."
}
```

# 2. Conversation (Multi-Modal)

```json
{
  "conversations": [
    [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "value": "You are a helpful AI assistant."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "value": "Can you describe this image?"
          },
          {
            "type": "image",
            "value": "/path/to/file"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "value": "The image shows a beautiful sunset over the mountains."
          }
        ]
      },
      // Additional Messages
    ],
    // Additional Conversations
  ]
}
```

# 3. Bounding Box (Image Specific)

```json
{
  "bbox": [
    {
      "label": "object_label",
      "coordinates": [x1, y1, x2, y2]
    }
  ]
}
```

Where:

- **label**: A string representing the class, category or description of the object.
- **coordinates**: An array of four numbers [x1, y1, x2, y2], representing the normalized top-left (x1, y1) and bottom-right (x2, y2) coordinates of the bounding box.

Ensure that the coordinates are normalized between 0 and 1, with (0,0) representing the top-left corner of the image.

# 4. Tabular Data

```json
{
  "tables": [
    [
      {
        "column1": "value1",
        "column2": "value2",
        // additional columns as needed
      },
      {
        "column1": "value3",
        "column2": "value4",
        // additional columns as needed
      }
      // additional rows as needed
    ],
    // additional tables as needed
  ]
}
```
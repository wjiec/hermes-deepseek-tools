# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/wjiec/hermes-deepseek-tools/security/advisories/new)
rather than a public issue. Include the plugin version, Hermes version, and steps to reproduce.
You should receive a response within 7 days.

## Scope

In scope: anything that lets this plugin

- send a profile's `DEEPSEEK_API_KEY` anywhere other than `api.deepseek.com`, or use one profile's key
  for another profile's request on a multiplexed gateway;
- upload a local file through `deepseek_ocr` that Hermes' own file tools would refuse to read, or a file
  outside the active profile's media cache on a multiplexed gateway;
- leak keys or request bodies into logs or error messages.

Issues in Hermes Agent itself belong upstream at
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent/security).

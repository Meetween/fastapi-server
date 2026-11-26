# Changelog

Notable changes to the Meetween FastAPI server. Newest first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

From 0.2.0 onward this file is written by [release-please](https://github.com/googleapis/release-please)
from the pull request titles merged into `main`. Entries are not added by hand. Unreleased work is
visible in the open `chore(main): release ...` pull request; merging it commits the version, creates
the `vX.Y.Z` tag, publishes the GitHub Release and deploys it to the VM.

## [0.5.0](https://github.com/Meetween/fastapi-server/compare/v0.4.1...v0.5.0) (2026-11-26)


### Added

* **agentar:** accounts, profile, files, meetings, lifecycle, prompt, report and events ([#147](https://github.com/Meetween/fastapi-server/issues/147)) ([f7adbcf](https://github.com/Meetween/fastapi-server/commit/f7adbcf5c796a6a76911da726e511c25e77e3fb5))
* **agentar:** settings, data model, contract and storage foundation ([#146](https://github.com/Meetween/fastapi-server/issues/146)) ([76c66b0](https://github.com/Meetween/fastapi-server/commit/76c66b056c0a0408dcc5659d13a501b83fb1c28e))
* **auth:** introduce service tokens and end of meeting signal ([#118](https://github.com/Meetween/fastapi-server/issues/118)) ([bea458e](https://github.com/Meetween/fastapi-server/commit/bea458e744a9b11b28cec162fac4f39257e10df7))
* **vp:** add the virtual participant control route and the worker key on chat ([#144](https://github.com/Meetween/fastapi-server/issues/144)) ([45be453](https://github.com/Meetween/fastapi-server/commit/45be453e378af3704130fecfc03d505b2972d918))
* **websocket:** utterance transcripts, ASR guards and grounded virtual participant answers ([#145](https://github.com/Meetween/fastapi-server/issues/145)) ([f26383a](https://github.com/Meetween/fastapi-server/commit/f26383a9c1db125aa1da63bf13c17473100b16e2))


### Fixed

* **agentar:** mount the Agentar routes and the lifecycle tick only with AGENTAR_ENABLED ([#150](https://github.com/Meetween/fastapi-server/issues/150)) ([8842755](https://github.com/Meetween/fastapi-server/commit/8842755d3278fba84eea1b889cc75047d85933d5))
* **agentar:** wire the modules and apply the showcase fixes ([#149](https://github.com/Meetween/fastapi-server/issues/149)) ([d49b93d](https://github.com/Meetween/fastapi-server/commit/d49b93d29cab496a833545bfb4fb22da558e785d))

## [0.4.1](https://github.com/Meetween/fastapi-server/compare/v0.4.0...v0.4.1) (2026-10-15)


### Fixed

* **auth:** tell clients when the access token has expired ([#94](https://github.com/Meetween/fastapi-server/issues/94)) ([950bde7](https://github.com/Meetween/fastapi-server/commit/950bde7af02d8010c61128e0ab1caa1e44b6765e))

## [0.4.0](https://github.com/Meetween/fastapi-server/compare/v0.3.0...v0.4.0) (2026-10-15)


### Added

* **health:** report the commit the service was built from ([#115](https://github.com/Meetween/fastapi-server/issues/115)) ([479c550](https://github.com/Meetween/fastapi-server/commit/479c550c12f2932c13d0ea36276f84ce28c897ad))


### Fixed

* **deploy:** keep container output off the system journal ([#110](https://github.com/Meetween/fastapi-server/issues/110)) ([13e90d7](https://github.com/Meetween/fastapi-server/commit/13e90d7a9c55029f610b3edae17e182b96bffe37))

## [0.3.0](https://github.com/Meetween/fastapi-server/compare/v0.2.0...v0.3.0) (2026-10-15)


### Added

* **deploy:** ship the service as a container to dev and prod ([#105](https://github.com/Meetween/fastapi-server/issues/105)) ([56e7e74](https://github.com/Meetween/fastapi-server/commit/56e7e749c817c486ca65a8f40b283f1023c08be5))
* **participants:** Receive and store participant name  ([#92](https://github.com/Meetween/fastapi-server/issues/92)) ([f8199f7](https://github.com/Meetween/fastapi-server/commit/f8199f7a7da13e8c186adb291141b6bf3ab3e29f))


### Fixed

* **deploy:** accept registry tokens longer than 255 characters ([#108](https://github.com/Meetween/fastapi-server/issues/108)) ([b8563ee](https://github.com/Meetween/fastapi-server/commit/b8563ee7675718f4ac997a2453b240447b64df06))
* **deploy:** keep the SSH session alive during a long pull ([#109](https://github.com/Meetween/fastapi-server/issues/109)) ([e05b158](https://github.com/Meetween/fastapi-server/commit/e05b158f440277d513dc53b74b1f1cd5c0999aa2))
* **deploy:** pull the image with the deploy job token ([#107](https://github.com/Meetween/fastapi-server/issues/107)) ([eabbfa1](https://github.com/Meetween/fastapi-server/commit/eabbfa1f58c3d28a1c65d9ff3a0aa19664274ecc))

## [0.2.0](https://github.com/Meetween/fastapi-server/compare/v0.1.0...v0.2.0) (2026-09-23)


### Fixed

* **deps:** declare pymongo and drop the unused pytest-mock ([#97](https://github.com/Meetween/fastapi-server/issues/97)) ([4fc9c07](https://github.com/Meetween/fastapi-server/commit/4fc9c0720c5874341b7fdddc625197f90330184c))

## [0.1.0] (2026-09-22)

The service as it stood when release automation started: live captions, transcription, translation,
summaries, catch up, chat, agenda tracking, JWT authentication and recording lifecycle, developed on
`develop` since 2025-11-13 without a version bump, a tag or a changelog. The tag `v0.1.0` marks this
state. The earlier tag `restore-point/fastapi-ac7a495` is a recovery point, not a release.

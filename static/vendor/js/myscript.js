// ============      PROFIL   ==============

const flashDataI = $('.flash-profil').data('flashdata');

if (flashDataI) {
	Swal.fire({
		icon: 'success',
		title: 'Profil',
		text: 'Berhasil ' + flashDataI
	});
}

// ============      PASSWORD  ==============

const flashDataJ = $('.flash-password').data('flashdata');

if (flashDataJ) {
	Swal.fire({
		icon: 'success',
		title: 'Password',
		text: 'Berhasil ' + flashDataJ
	});
}

// ============      MEMBER    ==============

const flashDataA = $('.flash-member').data('flashdata');

if (flashDataA) {
	Swal.fire({
		icon: 'success',
		title: 'Member ',
		text: 'Berhasil ' + flashDataA
	});
}

// hapus-member
$('.hapus-member').on('click', function (e) {

	e.preventDefault();
	const href = $(this).attr('href');

	Swal.fire({
		icon: 'warning',
		title: 'Apakah anda yakin',
		text: "Artikel akan dihapus",
		showCancelButton: true,
		confirmButtonColor: '#3085d6',
		cancelButtonColor: '#d33',
		confirmButtonText: 'Hapus Data!'
	}).then((result) => {
		if (result.value) {
			document.location.href = href;
		}
	})

});


// ============      INFORMASI PUBLIK    ==============

const flashDataB = $('.flash-informasi').data('flashdata');

if (flashDataB) {
	Swal.fire({
		icon: 'success',
		title: 'Informasi ',
		text: 'Berhasil ' + flashDataB,
	});
}

// hapus-informasi
$('.hapus-informasi').on('click', function (e) {

	e.preventDefault();
	const href = $(this).attr('href');

	Swal.fire({
		icon: 'warning',
		title: 'Apakah anda yakin',
		text: "Informasi akan dihapus",
		showCancelButton: true,
		confirmButtonColor: '#3085d6',
		cancelButtonColor: '#d33',
		confirmButtonText: 'Hapus Data!'
	}).then((result) => {
		if (result.value) {
			document.location.href = href;
		}
	})

});


// ============      KATEGORI    ==============

const flashDataD = $('.flash-kategori').data('flashdata');

if (flashDataD) {
	Swal.fire({
		icon: 'success',
		title: 'Kategori ',
		text: 'Berhasil ' + flashDataD,
	});
}

// hapus-kategori
$('.hapus-kategori').on('click', function (e) {

	e.preventDefault();
	const href = $(this).attr('href');

	Swal.fire({
		icon: 'warning',
		title: 'Apakah anda yakin',
		text: "Kategori akan dihapus",
		showCancelButton: true,
		confirmButtonColor: '#3085d6',
		cancelButtonColor: '#d33',
		confirmButtonText: 'Hapus Data!'
	}).then((result) => {
		if (result.value) {
			document.location.href = href;
		}
	})

});

// ============      SKPD    ==============

const flashDataE = $('.flash-skpd').data('flashdata');

if (flashDataE) {
	Swal.fire({
		icon: 'success',
		title: 'SKPD ',
		text: 'Berhasil ' + flashDataE
	});
}

// hapus-skpd
$('.hapus-skpd').on('click', function (e) {

	e.preventDefault();
	const href = $(this).attr('href');

	Swal.fire({
		icon: 'warning',
		title: 'Apakah anda yakin',
		text: "SKPD akan dihapus",
		showCancelButton: true,
		confirmButtonColor: '#3085d6',
		cancelButtonColor: '#d33',
		confirmButtonText: 'Hapus Data!'
	}).then((result) => {
		if (result.value) {
			document.location.href = href;
		}
	})

});

// ============      BERITA    ==============

const flashDataF = $('.flash-berita').data('flashdata');

if (flashDataF) {
	Swal.fire({
		icon: 'success',
		title: 'Berita ',
		text: 'Berhasil ' + flashDataF,
	});
}

// hapus-berita
$('.hapus-berita').on('click', function (e) {

	e.preventDefault();
	const href = $(this).attr('href');

	Swal.fire({
		icon: 'warning',
		title: 'Apakah anda yakin',
		text: "Berita akan dihapus",
		showCancelButton: true,
		confirmButtonColor: '#3085d6',
		cancelButtonColor: '#d33',
		confirmButtonText: 'Hapus Data!'
	}).then((result) => {
		if (result.value) {
			document.location.href = href;
		}
	})

});

// ============      KOTAK SARAN   ==============

const flashDataG = $('.flash-pesan').data('flashdata');

if (flashDataG) {
	Swal.fire({
		icon: 'success',
		title: 'Kotak Saran ',
		text: 'Berhasil ' + flashDataG
	});
}

// hapus-pesan
$('.hapus-pesan').on('click', function (e) {

	e.preventDefault();
	const href = $(this).attr('href');

	Swal.fire({
		icon: 'warning',
		title: 'Apakah anda yakin',
		text: "Pesan akan dihapus",
		showCancelButton: true,
		confirmButtonColor: '#3085d6',
		cancelButtonColor: '#d33',
		confirmButtonText: 'Hapus Data!'
	}).then((result) => {
		if (result.value) {
			document.location.href = href;
		}
	})

});

// ============      PERMOHONAN   ==============

const flashDataH = $('.flash-permohonan').data('flashdata');

if (flashDataH) {
	Swal.fire({
		icon: 'success',
		title: 'Permohonan ',
		text: 'Berhasil ' + flashDataH,
	});
}

// hapus-permohonan
$('.hapus-permohonan').on('click', function (e) {

	e.preventDefault();
	const href = $(this).attr('href');

	Swal.fire({
		icon: 'warning',
		title: 'Apakah anda yakin',
		text: "Permohonan akan dihapus",
		showCancelButton: true,
		confirmButtonColor: '#3085d6',
		cancelButtonColor: '#d33',
		confirmButtonText: 'Hapus Data!'
	}).then((result) => {
		if (result.value) {
			document.location.href = href;
		}
	})

});

// ============      DOKKUMEN INFORMASI   ==============

const flashDataK = $('.flash-dokumen').data('flashdata');

if (flashDataK) {
	Swal.fire({
		icon: 'success',
		title: 'Dokumen Informasi ',
		text: 'Berhasil ' + flashDataK,
	});
};
